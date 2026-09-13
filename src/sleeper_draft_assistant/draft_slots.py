"""Roster assignment shared by draft valuation and simulation."""
from __future__ import annotations

from collections.abc import Iterable


SLOT_POSITIONS = {
    **{position: frozenset({position}) for position in ("QB", "RB", "WR", "TE", "K", "DEF")},
    "FLEX": frozenset({"RB", "WR", "TE"}),
    "SUPER_FLEX": frozenset({"QB", "RB", "WR", "TE"}),
    "REC_FLEX": frozenset({"WR", "TE"}),
    "WRRB_FLEX": frozenset({"RB", "WR"}),
}
NON_STARTERS = frozenset({"BN", "IR", "TAXI"})


def starting_slots(positions: Iterable[str]) -> tuple[str, ...]:
    slots = tuple(position for position in positions if position not in NON_STARTERS)
    unsupported = set(slots) - SLOT_POSITIONS.keys()
    if unsupported:
        raise ValueError(f"Unsupported draft roster positions: {', '.join(sorted(unsupported))}")
    return slots


def assign_starters(
    candidates: Iterable[tuple[str, str, float]], slots: Iterable[str]
) -> dict[int, str]:
    """Return the maximum-weight eligible starter set, one player per slot.

    Greedy weight order with augmenting paths is exact for this matching
    problem: an accepted player can move slots, so an early flexible assignment
    cannot block a later player with more restrictive eligibility.
    """
    slot_eligibility = [SLOT_POSITIONS[slot] for slot in slots]
    assignments: dict[int, str] = {}
    positions: dict[str, str] = {}

    def place(player_id: str, visited: set[int]) -> bool:
        for slot, eligible in enumerate(slot_eligibility):
            if slot in visited or positions[player_id] not in eligible:
                continue
            visited.add(slot)
            occupant = assignments.get(slot)
            if occupant is None or place(occupant, visited):
                assignments[slot] = player_id
                return True
        return False

    for player_id, position, points in sorted(candidates, key=lambda row: row[2], reverse=True):
        if player_id in positions or points < 0:
            continue
        positions[player_id] = position
        place(player_id, set())
        if len(assignments) == len(slot_eligibility):
            break
    return assignments
