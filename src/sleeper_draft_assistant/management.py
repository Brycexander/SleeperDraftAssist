"""Read-only, current-roster advice, independent of draft initialization."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests

from .cache import cache_directory
from .models import normalize_position
from .sleeper import SleeperClient
from .strategy import WeeklyPlayer, optimize_lineup, suggest_trades, suggest_waivers
from .weekly import load_weekly_players


class TeamService:
    def __init__(
        self,
        client: SleeperClient | None = None,
        loader: Callable[..., Any] = load_weekly_players,
        cache_dir: Path | None = None,
    ) -> None:
        self.client = client or SleeperClient()
        self.loader = loader
        self.cache_dir = cache_dir

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
    ) -> dict[str, Any]:
        if action not in {"lineup", "waivers", "trades"}:
            raise ValueError("Choose lineup, waivers, or trades.")
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
        settings = league.get("settings") or {}
        if settings.get("best_ball"):
            warnings.append("This is a best-ball league: Sleeper selects the scoring lineup automatically.")
        if action != "lineup" and (settings.get("type") == 2 or settings.get("max_keepers")):
            warnings.append("Keeper and dynasty rights, draft picks, contracts, and player age are not valued in these roster moves.")
        if action != "lineup" and any(p.roster_value <= 0 for p in active):
            warnings.append("Players without season-strength estimates are protected from drops and trades; missing estimates can reduce the suggestions shown.")
        suggestions: list[dict[str, Any]] = []
        capacity = len([p for p in positions if p not in {"IR", "TAXI"}])
        if action == "waivers":
            if settings.get("disable_adds"):
                warnings.append("Player additions are disabled in this league.")
            else:
                suggestions = suggest_waivers(players, rosters, int(roster["roster_id"]), slots, limit=limit, mode=mode, capacity=capacity)
            warnings.append("Unrostered players may still need a waiver claim. Check claim timing, FAAB, and roster eligibility in Sleeper.")
        elif action == "trades":
            deadline = int(settings.get("trade_deadline") or 0)
            if settings.get("disable_trades") or (deadline and current_week > deadline):
                warnings.append("Trading is disabled or the league's trade deadline has passed.")
            else:
                suggestions = suggest_trades(players, rosters, int(roster["roster_id"]), slots, limit=limit, mode=mode, capacity=capacity)
            warnings.append("Trade gains are estimates for this week's roster. Review longer-term value and processing time; mutual improvement does not predict acceptance.")
        return {
            "action": action, "league_id": resolved_id, "league_name": league["name"].strip(),
            "season": season, "week": week, "mode": mode, "roster_id": roster["roster_id"],
            "scoring": {"ppr": scoring.get("rec", 0), "passing_td": scoring.get("pass_td", 0)},
            "slots": slots, "lineup": lineup, "suggestions": suggestions,
            "warnings": list(dict.fromkeys(warnings)), "sources": data.sources,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "method": (
                "Maximum projected points across every legal starter assignment."
                if mode == "projection" else
                "Tiers guide choices within positions; projections put FLEX choices on a common scale."
            ),
        }
