"""Bounded waiver and trade searches over expert ROS rank credits."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .expert_data import ExpertBoard
from .models import normalize_position
from .roster_value import RosterScore, compare_rosters, evaluate_roster
from .strategy import WeeklyPlayer, _active_ids, _eligible, _find_roster, _movable, _NONSTARTER


@dataclass(frozen=True)
class MoveSearch:
    suggestions: list[dict[str, Any]]
    discussion_candidates: list[dict[str, Any]]
    near_misses: list[dict[str, Any]]
    diagnostics: dict[str, Any]


def _supported_slots(slots: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalize_position(s) for s in slots if s.upper() not in _NONSTARTER and normalize_position(s) in {"QB", "RB", "WR", "TE", "FLEX", "SUPER_FLEX", "REC_FLEX", "WRRB_FLEX"})


def _eligible_player(p: WeeklyPlayer, board: ExpertBoard, slots: tuple[str, ...]) -> bool:
    return p.player_id in board.values and p.position in {"QB", "RB", "WR", "TE"} and p.rosterable and any(_eligible(p, slot) for slot in slots)


def _row(p: WeeklyPlayer, board: ExpertBoard) -> dict[str, Any]:
    value = board.values[p.player_id]
    return {"player_id": p.player_id, "name": p.name, "position": p.position, "team": p.team,
            "median_rank": value.median_rank, "positional_rank": value.positional_rank,
            "rank_credit": value.rank_credit, "best_rank": value.best_rank,
            "worst_rank": value.worst_rank, "rank_spread": value.rank_spread,
            "agreement": value.agreement, "contributors": value.contributors,
            "status": p.status, "bye": p.bye, "locked": p.locked}


def _scores(score: RosterScore) -> dict[str, Any]:
    return {"starters": score.starters, "backup1": score.backup1, "backup2": score.backup2, "surplus": score.surplus,
            "missing_ids": list(score.missing_ids), "empty_slots": score.empty_slots}


def _better(before: RosterScore, after: RosterScore) -> bool:
    return not set(after.missing_ids) - set(before.missing_ids) and after.empty_slots <= before.empty_slots and compare_rosters(before, after) > 0


def _packages(pool: list[WeeklyPlayer], starters: set[str]) -> list[tuple[WeeklyPlayer, ...]]:
    singles = [(p,) for p in pool]
    pairs = list(combinations(pool, 2))
    pairs.sort(key=lambda pair: (pair[0].position != pair[1].position,
                                 (pair[0].player_id in starters) != (pair[1].player_id in starters),
                                 tuple(sorted(p.player_id for p in pair))), reverse=True)
    # Keep deterministic, bounded packages; rank credits are used by callers.
    return singles + pairs[:32]


def suggest_expert_waivers(players: dict[str, WeeklyPlayer], rosters: list[dict[str, Any]], roster_id: int,
                           slots: tuple[str, ...], board: ExpertBoard, *, capacity: int, limit: int = 10) -> MoveSearch:
    own = _find_roster(rosters, roster_id)
    active = _active_ids(own)
    occupied = set().union(*(set(str(pid) for key in ("players", "reserve", "taxi", "starters") for pid in roster.get(key) or []) for roster in rosters))
    slots = _supported_slots(slots)
    ranked = [p for p in players.values() if p.player_id not in occupied and _eligible_player(p, board, slots) and not p.locked]
    drops = [None] if len(active) < capacity else [players[pid] for pid in sorted(active) if pid in players and _movable(players[pid]) and players[pid].player_id in board.values]
    before = evaluate_roster(active, players, slots, board)
    suggestions, rejects = [], {}
    for add in sorted(ranked, key=lambda p: (board.values[p.player_id].rank_credit, p.player_id), reverse=True):
        best = None
        for drop in drops:
            ids = (active - ({drop.player_id} if drop else set())) | {add.player_id}
            after = evaluate_roster(ids, players, slots, board)
            if not _better(before, after):
                rejects["no_own_improvement"] = rejects.get("no_own_improvement", 0) + 1
                continue
            item = {"kind": "expert_waiver", "add": _row(add, board), "drop": _row(drop, board) if drop else None,
                    "own_before": _scores(before), "own_after": _scores(after),
                    "own_gain": {key: round(getattr(after, key) - getattr(before, key), 6) for key in ("starters", "backup1", "backup2", "surplus")},
                    "reason": "Improves the legal long-term roster assignment under the selected expert ROS ranks.",
                    "warnings": ["Rank credit is an ordering heuristic, not projected fantasy points or a waiver guarantee."],
                    "search_scope": "All eligible ranked free agents and every legal ranked drop were evaluated."}
            if best is None or tuple(item["own_gain"].values()) > tuple(best["own_gain"].values()):
                best = item
        if best:
            suggestions.append(best)
    suggestions.sort(key=lambda x: tuple(x["own_gain"].values()), reverse=True)
    return MoveSearch(suggestions[:limit], [], [], {"candidates_evaluated": len(ranked) * len(drops), "eligible_own_players": len(active), "rejections": rejects, "search_scope": "All eligible ranked free agents and legal drops"})


def suggest_expert_trades(players: dict[str, WeeklyPlayer], rosters: list[dict[str, Any]], roster_id: int,
                          slots: tuple[str, ...], board: ExpertBoard, *, capacity: int, approach: str = "balanced",
                          form: dict[str, dict[str, Any]] | None = None, limit: int = 10) -> MoveSearch:
    own = _find_roster(rosters, roster_id)
    own_ids = _active_ids(own)
    slots = _supported_slots(slots)
    own_pool = [p for p in (players[pid] for pid in own_ids if pid in players) if _eligible_player(p, board, slots) and _movable(p)]
    own_before = evaluate_roster(own_ids, players, slots, board)
    suggestions, discussions, near = [], [], []
    diagnostics = {"candidates_evaluated": 0, "eligible_own_players": len(own_pool), "rejections": {}}
    for other in rosters:
        if str(other.get("roster_id")) == str(roster_id):
            continue
        other_ids = _active_ids(other)
        other_pool = [p for p in (players[pid] for pid in other_ids if pid in players) if _eligible_player(p, board, slots) and _movable(p)]
        their_before = evaluate_roster(other_ids, players, slots, board)
        for give in [(p,) for p in own_pool] + list(combinations(own_pool, 2))[:32]:
            for receive in [(p,) for p in other_pool] + list(combinations(other_pool, 2))[:32]:
                diagnostics["candidates_evaluated"] += 1
                if len(give) != len(receive):
                    continue
                given = sum(board.values[p.player_id].rank_credit for p in give)
                received = sum(board.values[p.player_id].rank_credit for p in receive)
                if min(given, received) <= 0 or min(given, received) / max(given, received) < .75:
                    diagnostics["rejections"]["package_imbalance"] = diagnostics["rejections"].get("package_imbalance", 0) + 1
                    continue
                after_own_ids = (own_ids - {p.player_id for p in give}) | {p.player_id for p in receive}
                after_other_ids = (other_ids - {p.player_id for p in receive}) | {p.player_id for p in give}
                own_after = evaluate_roster(after_own_ids, players, slots, board)
                their_after = evaluate_roster(after_other_ids, players, slots, board)
                if not _better(own_before, own_after):
                    diagnostics["rejections"]["no_own_improvement"] = diagnostics["rejections"].get("no_own_improvement", 0) + 1
                    continue
                balanced = _better(their_before, their_after)
                row = {"partner_roster_id": other["roster_id"], "give": [_row(p, board) for p in give], "receive": [_row(p, board) for p in receive],
                       "own_before": _scores(own_before), "own_after": _scores(own_after), "partner_before": _scores(their_before), "partner_after": _scores(their_after),
                       "own_gain": {key: round(getattr(own_after, key) - getattr(own_before, key), 6) for key in ("starters", "backup1", "backup2", "surplus")},
                       "partner_gain": {key: round(getattr(their_after, key) - getattr(their_before, key), 6) for key in ("starters", "backup1", "backup2", "surplus")},
                       "value_ratio": round(received / given, 4), "reason": "Compared legal roster assignments using selected expert rest-of-season rank credits.",
                       "warnings": ["Rank credit is a transparent ordering heuristic, not projected points or an acceptance probability."],
                       "search_scope": "All eligible one-for-one exchanges and bounded two-player packages were evaluated."}
                if approach == "balanced" and balanced:
                    row["kind"] = "expert_balanced_trade"; suggestions.append(row)
                elif approach == "opportunities" and form:
                    signals = [form.get(p.player_id, {}).get("signal") for p in give + receive]
                    if "sell_high" in signals and "buy_low" in signals and received > given and not balanced:
                        row["kind"] = "expert_opportunity"; discussions.append(row)
                    elif balanced:
                        row["kind"] = "expert_balanced_trade"; suggestions.append(row)
                    else:
                        diagnostics["rejections"]["no_partner_improvement"] = diagnostics["rejections"].get("no_partner_improvement", 0) + 1
    key = lambda x: tuple(x["own_gain"].values())
    suggestions.sort(key=key, reverse=True); discussions.sort(key=key, reverse=True)
    return MoveSearch(suggestions[:limit], discussions[:limit], near[:3], diagnostics)
