from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


FLEX_POSITIONS = frozenset({"RB", "WR", "TE"})


def normalize_position(position: str | None) -> str:
    value = (position or "").upper()
    if value in {"D/ST", "DST"}:
        return "DEF"
    return value


@dataclass(frozen=True, slots=True)
class Player:
    sleeper_id: str
    name: str
    position: str
    team: str
    ecr: float
    uncertainty: float
    bye: int | None = None
    projected_points: float = 0.0
    vorp: float = 0.0
    value_score: float = 0.0
    value_rank: float = 999.0
    adp: float | None = None
    adp_uncertainty: float = 12.0
    injury_risk: float = 0.0
    role_risk: float = 0.0
    outcome_cv: float = 0.25
    projection_source: str = "ECR fallback"


@dataclass(frozen=True, slots=True)
class LeagueRules:
    teams: int
    rounds: int
    roster_positions: tuple[str, ...]
    scoring: dict[str, float]

    @property
    def starters(self) -> tuple[str, ...]:
        return tuple(p for p in self.roster_positions if p != "BN")

    @property
    def bench_slots(self) -> int:
        return self.roster_positions.count("BN")

    def required_count(self, position: str) -> int:
        return self.starters.count(position)

    @property
    def flex_slots(self) -> int:
        return self.starters.count("FLEX")


@dataclass(slots=True)
class LeagueContext:
    league_id: str
    username: str
    user_id: str
    roster_id: int
    draft_slot: int
    league: dict[str, Any]
    draft: dict[str, Any]
    picks: list[dict[str, Any]]
    traded_picks: list[dict[str, Any]]
    users: list[dict[str, Any]]
    rosters: list[dict[str, Any]]
    rules: LeagueRules

    @property
    def roster_to_user(self) -> dict[int, str]:
        slot_to_roster = {
            int(slot): int(roster_id)
            for slot, roster_id in self.draft.get("slot_to_roster_id", {}).items()
        }
        return {
            roster_id: user_id
            for user_id, slot in self.draft.get("draft_order", {}).items()
            if (roster_id := slot_to_roster.get(int(slot))) is not None
        }


@dataclass(slots=True)
class DraftState:
    next_pick: int
    rosters: dict[int, list[str]] = field(default_factory=dict)
    drafted: set[str] = field(default_factory=set)

    def clone(self) -> DraftState:
        return DraftState(
            next_pick=self.next_pick,
            rosters={roster_id: list(players) for roster_id, players in self.rosters.items()},
            drafted=set(self.drafted),
        )


@dataclass(frozen=True, slots=True)
class Recommendation:
    player: Player
    availability_rate: float
    selection_rate: float
    mean_score: float
    top_roster_rate: float
    samples: int
