"""Roster evaluation using expert rank credits, independent of forecasts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .expert_data import ExpertBoard
from .models import normalize_position
from .strategy import WeeklyPlayer, _assignment, _eligible, _NONSTARTER


@dataclass(frozen=True)
class RosterScore:
    starters: float
    backup1: float
    backup2: float
    surplus: float
    missing_ids: tuple[str, ...] = ()
    empty_slots: int = 0

    def key(self) -> tuple[float, float, float, float]:
        return tuple(round(v, 6) for v in (self.starters, self.backup1, self.backup2, self.surplus))


def compare_rosters(before: RosterScore, after: RosterScore) -> int:
    return (after.key() > before.key()) - (after.key() < before.key())


def _slots(slots: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(normalize_position(slot) for slot in slots if slot.upper() not in _NONSTARTER)


def evaluate_roster(ids: set[str], players: dict[str, WeeklyPlayer], slots: tuple[str, ...], board: ExpertBoard) -> RosterScore:
    slots = _slots(slots)
    ranked = [players[pid] for pid in sorted(ids) if pid in players and pid in board.values and players[pid].rosterable]
    missing = tuple(sorted(pid for pid in ids if pid in players and players[pid].position in {"QB", "RB", "WR", "TE"} and pid not in board.values))
    if not slots:
        return RosterScore(0.0, 0.0, 0.0, sum(board.values[p.player_id].rank_credit for p in ranked), missing, 0)
    # Assignment rows are slots and columns are ranked players plus empty choices.
    matrix = []
    for slot in slots:
        matrix.append([board.values[p.player_id].rank_credit if _eligible(p, slot) else -1e12 for p in ranked] + [0.0] * len(slots))
    chosen = _assignment(matrix)
    used: set[str] = set()
    starters = 0.0
    for row, col in enumerate(chosen):
        if 0 <= col < len(ranked) and matrix[row][col] > -1e11:
            used.add(ranked[col].player_id)
            starters += board.values[ranked[col].player_id].rank_credit
    # Backups are deliberately small, separate layers. A player is counted once.
    remaining = [p for p in ranked if p.player_id not in used]
    layers: list[float] = []
    for layer in range(2):
        values = []
        covered: set[str] = set()
        for position in {p.position for p in remaining}:
            eligible = [p for p in remaining if p.player_id not in covered and p.position == position and any(_eligible(p, slot) for slot in slots)]
            if eligible:
                best = max(eligible, key=lambda p: (board.values[p.player_id].rank_credit, p.player_id))
                values.append(board.values[best.player_id].rank_credit)
                covered.add(best.player_id)
        layers.append(sum(values))
        remaining = [p for p in remaining if p.player_id not in covered]
    surplus = sum(board.values[p.player_id].rank_credit for p in remaining)
    empty = max(0, len(slots) - len(used))
    return RosterScore(round(starters, 6), round(layers[0], 6), round(layers[1], 6), round(surplus, 6), missing, empty)


def replacement_ranks(players: dict[str, WeeklyPlayer], rosters: list[dict[str, Any]], board: ExpertBoard) -> dict[str, float | None]:
    owned = set().union(*(str(pid) for roster in rosters for key in ("players", "reserve", "taxi", "starters") for pid in roster.get(key) or []))
    answer: dict[str, float | None] = {}
    for position in ("QB", "RB", "WR", "TE"):
        values = [v for pid, v in board.values.items() if v.position == position and pid not in owned]
        answer[position] = min((v.positional_rank for v in values), default=None)
    return answer
