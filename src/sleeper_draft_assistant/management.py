"""Read-only, current-roster advice, independent of draft initialization."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from .cache import cache_directory
from .models import normalize_position
from .sleeper import SleeperClient
from .strategy import WeeklyPlayer, optimize_lineup, suggest_trades, suggest_waivers, suggest_opportunity_trades
from .trade_history import load_trade_history
from .weekly import load_expert_players, load_weekly_players
from .expert_sources import load_expert_board
from .expert_moves import suggest_expert_trades, suggest_expert_waivers
from .tier_charts import load_ppr_tier_charts


class TeamService:
    def __init__(
        self,
        client: SleeperClient | None = None,
        loader: Callable[..., Any] = load_weekly_players,
        cache_dir: Path | None = None,
        expert_loader: Callable[..., Any] = load_expert_players,
    ) -> None:
        self.client = client or SleeperClient()
        self.loader = loader
        self.cache_dir = cache_dir
        self.expert_loader = expert_loader

    def _current_league(
        self, league_id: str, user_id: str, season: int
    ) -> tuple[dict[str, Any], list[str]]:
        league = self.client.league(league_id)
        if not league:
            raise ValueError("Sleeper could not find that league.")
        if int(league["season"]) == season:
            return league, []
        # Only follow an explicit renewal chain, never a similar league name.
        matches = []
        for candidate in self.client.user_leagues(user_id, season):
            ancestor = candidate
            for _ in range(5):
                previous = str(ancestor.get("previous_league_id") or "")
                if previous == league_id:
                    matches.append(candidate)
                    break
                if not previous:
                    break
                ancestor = self.client.league(previous) or {}
        if len(matches) != 1:
            raise ValueError(
                f"This league is from {league['season']}; choose its {season} league ID "
                "to get advice for current rosters."
            )
        return matches[0], [f"Using the renewed {season} league for this saved league."]

    def advise(
        self,
        league_id: str,
        username: str,
        action: str = "lineup",
        week: int | None = None,
        mode: str = "projection",
        limit: int = 10,
        refresh: bool = False,
        confirm_tiers_week: bool = False,
        expected_season: int | None = None,
        trade_approach: str = "balanced",
        valuation_source: str = "sleeper",
        show_ppr_tiers: bool = False,
    ) -> dict[str, Any]:
        if action not in {"lineup", "waivers", "trades"}:
            raise ValueError("Choose lineup, waivers, or trades.")
        if trade_approach not in {"balanced", "opportunities"} or (trade_approach != "balanced" and action != "trades"):
            raise ValueError("Buy-low/sell-high mode is available for trades only; choose balanced or opportunities.")
        if valuation_source not in {"sleeper", "experts"}:
            raise ValueError("valuation_source must be sleeper or experts")
        if mode not in {"projection", "tiers"}:
            raise ValueError("Choose projection or tiers mode.")
        if not 1 <= limit <= 25:
            raise ValueError("The result limit must be between 1 and 25.")
        state = self.client.nfl_state()
        season = int(state["season"])
        if expected_season is not None and expected_season != season:
            raise ValueError(f"Sleeper is in the {season} season; refresh the season and week selection.")
        if confirm_tiers_week and (week is None or expected_season is None):
            raise ValueError("Confirm tiers for an explicit season and week.")
        current_week = int(state.get("week") or 0)
        if state.get("season_type") != "regular" or not 1 <= current_week <= 18:
            raise ValueError("Weekly roster advice is available during the NFL regular season.")
        week = current_week if week is None else week
        if not current_week <= week <= 18:
            raise ValueError(f"Choose a week from {current_week} through 18; past weeks cannot use live rosters.")
        if action != "lineup" and week != current_week:
            raise ValueError(f"Waiver and trade advice uses the current week ({current_week}) to respect current game locks.")
        user = self.client.user(username)
        if not user or not user.get("user_id"):
            raise ValueError("Sleeper could not find that username.")
        user_id = str(user["user_id"])
        league, warnings = self._current_league(str(league_id), user_id, season)
        resolved_id = str(league["league_id"])
        rosters = [dict(roster) for roster in self.client.rosters(resolved_id)]
        owned = [
            roster for roster in rosters
            if str(roster.get("owner_id") or "") == user_id
            or user_id in {str(owner) for owner in roster.get("co_owners") or []}
        ]
        if len(owned) != 1:
            raise ValueError("The username must own or co-own exactly one roster in this league.")
        roster = owned[0]
        # Matchups preserve the selected week's slot order, including already locked starters.
        try:
            matchups = self.client.matchups(resolved_id, week)
            for entry in matchups or []:
                target = next((r for r in rosters if r["roster_id"] == entry["roster_id"]), None)
                if target is not None and "starters" in entry:
                    target["starters"] = list(entry["starters"] or [])
        except requests.RequestException:
            warnings.append("Weekly lineup sync failed; using Sleeper's current roster starters. Verify locked slots in Sleeper.")
        positions = tuple(normalize_position(p) for p in league["roster_positions"])
        slots = tuple(p for p in positions if p not in {"BN", "IR", "TAXI"})
        scoring = {k: float(v) for k, v in league["scoring_settings"].items()}
        data = self.loader(
            self.cache_dir or cache_directory(), season, week, scoring,
            refresh=refresh, confirm_tiers_week=confirm_tiers_week,
        )
        players = dict(data.players)
        warnings.extend(data.warnings)
        for team in rosters:
            for player_id in dict.fromkeys(
                str(p) for key in ("players", "reserve", "taxi", "starters")
                for p in team.get(key) or [] if p and str(p) != "0"
            ):
                if player_id not in players:
                    players[player_id] = WeeklyPlayer(
                        player_id=player_id, name=f"Unknown player ({player_id})",
                        position="", team="", points=None, rosterable=False,
                    )
                    warnings.append(f"Player {player_id} is missing from the player feed; roster advice may be incomplete.")
        inactive_ids = {str(p) for key in ("reserve", "taxi") for p in roster.get(key) or []}
        active = [players[str(p)] for p in roster.get("players") or [] if str(p) in players and str(p) not in inactive_ids]
        lineup = optimize_lineup(active, slots, mode=mode, current_starters=roster.get("starters") or [])
        warnings.extend(lineup.get("warnings", []))
        sources = list(data.sources)
        ppr_tiers: list[dict[str, Any]] = []
        ppr_tier_sources: list[dict[str, Any]] = []
        if show_ppr_tiers and action == "lineup":
            charts, chart_warnings, chart_sources = load_ppr_tier_charts(
                self.cache_dir or cache_directory(),
                season,
                week,
                players,
                {player.player_id for player in active},
                refresh=refresh,
            )
            ppr_tiers = charts
            ppr_tier_sources = chart_sources
            warnings.extend(chart_warnings)
            sources.extend(chart_sources)
        settings = league.get("settings") or {}
        if settings.get("best_ball"):
            warnings.append("This is a best-ball league: Sleeper selects the scoring lineup automatically.")
        if action != "lineup" and (settings.get("type") == 2 or settings.get("max_keepers")):
            warnings.append("Keeper and dynasty rights, draft picks, contracts, and player age are not valued in these roster moves.")
        if action != "lineup" and valuation_source == "sleeper" and any(p.roster_value <= 0 for p in active):
            warnings.append("Players without season-strength estimates are protected from drops and trades; missing estimates can reduce the suggestions shown.")
        suggestions: list[dict[str, Any]] = []
        trend_players: list[dict[str, Any]] = []
        discussion_candidates: list[dict[str, Any]] = []
        near_misses: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {}
        expert_status = None
        expert_panel: list[str] = []
        expert_selection: list[dict[str, Any]] = []
        expert_exclusions: list[dict[str, Any]] = []
        expert_coverage = 0
        replacement = {}
        capacity = len([p for p in positions if p not in {"IR", "TAXI"}])
        if valuation_source == "experts" and action in {"waivers", "trades"}:
            from .expert_moves import MoveSearch
            from .roster_value import replacement_ranks
            scoring_name = "PPR" if scoring.get("rec", 0) == 1 else "HALF" if scoring.get("rec", 0) == 0.5 else "STD"
            expert_status, board, expert_sources, expert_warnings = load_expert_board(
                self.cache_dir or cache_directory(), season=season, week=week, scoring=scoring_name,
                now=datetime.now(timezone.utc), refresh=refresh,
            )
            sources.extend(expert_sources)
            expert_exclusions = [
                item for source in expert_sources for item in source.get("excluded_experts", [])
            ]
            warnings.extend(expert_warnings)
            if board is not None and expert_status == "ready":
                expert_data = self.expert_loader(
                    self.cache_dir or cache_directory(), season, week, scoring, refresh=refresh
                )
                expert_players = dict(expert_data.players)
                warnings.extend(expert_data.warnings)
                sources.extend(expert_data.sources)
                for team in rosters:
                    for raw_id in dict.fromkeys(
                        str(p) for key in ("players", "reserve", "taxi", "starters")
                        for p in team.get(key) or [] if p and str(p) != "0"
                    ):
                        if raw_id not in expert_players:
                            value = board.values.get(raw_id)
                            expert_players[raw_id] = WeeklyPlayer(
                                player_id=raw_id, name=f"Unknown player ({raw_id})",
                                position=value.position if value else "", team="", points=None,
                                rosterable=value is not None, locked=True,
                                source="Missing from Sleeper metadata; conservatively locked",
                            )
                            warnings.append(f"Player {raw_id} is missing from Sleeper metadata and is conservatively locked in expert mode.")
                replacement = replacement_ranks(expert_players, rosters, board)
                if action == "waivers":
                    result = suggest_expert_waivers(expert_players, rosters, int(roster["roster_id"]), slots, board, capacity=capacity, limit=limit)
                else:
                    result = suggest_expert_trades(expert_players, rosters, int(roster["roster_id"]), slots, board, capacity=capacity, approach=trade_approach, limit=limit)
                suggestions = result.suggestions
                discussion_candidates = result.discussion_candidates
                near_misses = result.near_misses
                diagnostics = result.diagnostics
                expert_panel = list(board.panel_ids)
                expert_selection = [
                    {"expert_id": expert_id, "score": score}
                    for expert_id, score in board.panel_scores
                ]
                expert_coverage = len(board.values)
            else:
                warnings.append("Expert transaction advice is withheld until at least five experts qualify through the rolling multi-year ROS accuracy screen.")
            warnings.append("Expert transaction advice uses rank credits for rest-of-season roster comparison; it does not use Sleeper projections or Fantasy Football Tiers.")
        elif action == "waivers":
            if settings.get("disable_adds"):
                warnings.append("Player additions are disabled in this league.")
            else:
                suggestions = suggest_waivers(players, rosters, int(roster["roster_id"]), slots, limit=limit, mode=mode, capacity=capacity)
            warnings.append("Unrostered players may still need a waiver claim. Check claim timing, FAAB, and roster eligibility in Sleeper.")
        elif action == "trades":
            deadline = int(settings.get("trade_deadline") or 0)
            if settings.get("disable_trades") or (deadline and current_week > deadline):
                warnings.append("Trading is disabled or the league's trade deadline has passed.")
            elif trade_approach == "opportunities":
                rostered = {str(p) for r in rosters for p in r.get("players") or []}
                history = load_trade_history(self.cache_dir or cache_directory(), season, week, scoring,
                                             {pid: p for pid, p in players.items() if pid in rostered}, refresh=refresh)
                warnings.extend(history.warnings)
                sources.extend(history.sources)
                suggestions = suggest_opportunity_trades(players, rosters, int(roster["roster_id"]), slots,
                                                          history.trends, limit=limit, capacity=capacity)
                for team in rosters:
                    for pid in team.get("players") or []:
                        pid = str(pid)
                        if pid in history.trends and players[pid].position in {"QB", "RB", "WR", "TE"}:
                            trend_players.append({"player_id": pid, "name": players[pid].name,
                                                  "position": players[pid].position, "roster_id": team["roster_id"],
                                                  **history.trends[pid]})
            else:
                suggestions = suggest_trades(players, rosters, int(roster["roster_id"]), slots, limit=limit, mode=mode, capacity=capacity)
            warnings.append("Trade gains are estimates. Review longer-term value and processing time; modeled benefit does not predict acceptance.")
        return {
            "action": action, "league_id": resolved_id, "league_name": league["name"].strip(),
            "season": season, "week": week, "mode": mode, "roster_id": roster["roster_id"],
            "scoring": {"ppr": scoring.get("rec", 0), "passing_td": scoring.get("pass_td", 0)},
            "slots": slots, "lineup": lineup, "suggestions": suggestions,
            "ppr_tiers": ppr_tiers, "ppr_tier_sources": ppr_tier_sources,
            "warnings": list(dict.fromkeys(warnings)), "sources": sources,
            "trade_approach": trade_approach, "trend_players": trend_players,
            "valuation_source": valuation_source, "expert_status": expert_status,
            "expert_panel": expert_panel, "expert_selection": expert_selection,
            "expert_exclusions": expert_exclusions, "expert_coverage": expert_coverage,
            "discussion_candidates": discussion_candidates, "near_misses": near_misses,
            "diagnostics": diagnostics, "replacement_ranks": replacement,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "method": (
                "Expert rest-of-season roster comparison using selected expert rank credits."
                if valuation_source == "experts" and action in {"waivers", "trades"} else
                "Buy-low/sell-high: projected outlook and workload are compared with recent-form perception. Perception and confidence are heuristic, not observed market prices or acceptance odds."
                if action == "trades" and trade_approach == "opportunities" else
                "Maximum projected points across every legal starter assignment."
                if mode == "projection" else
                "Tiers guide choices within positions; projections put FLEX choices on a common scale."
            ),
        }
