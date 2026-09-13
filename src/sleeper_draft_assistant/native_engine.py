from __future__ import annotations

import math
import os
import warnings
from collections import Counter
from typing import Iterable

from .models import LeagueContext, Player, Recommendation
from .simulator import (
    MonteCarloDraft,
    SimulationReport,
    next_pick_for_roster,
    roster_for_pick,
)

try:
    from ._native import NativeDraftEngine as _NativeDraftEngine
except ImportError as exc:  # pragma: no cover - exercised with an import blocker
    _NativeDraftEngine = None
    _NATIVE_IMPORT_ERROR: ImportError | None = exc
else:
    _NATIVE_IMPORT_ERROR = None


POSITION_INDEX = {"QB": 0, "RB": 1, "WR": 2, "TE": 3, "K": 4, "DEF": 5}
POSITION_ORDER = tuple(POSITION_INDEX)


class NativeEngineUnavailable(RuntimeError):
    pass


def native_available() -> bool:
    return _NativeDraftEngine is not None


def native_import_error() -> ImportError | None:
    return _NATIVE_IMPORT_ERROR


def create_recommendation_engine(
    requested: str,
    context: LeagueContext,
    players: Iterable[Player],
    manager_biases: dict[str, dict[str, float]] | None,
    seed: int,
) -> tuple[MonteCarloDraft | NativeMonteCarloDraft, str]:
    player_list = list(players)
    if requested == "python":
        return MonteCarloDraft(context, player_list, manager_biases, seed), "python"
    if native_available():
        try:
            return NativeMonteCarloDraft(
                context, player_list, manager_biases, seed
            ), "cpp"
        except NativeEngineUnavailable as exc:
            message = str(exc)
    else:
        message = f"native C++ engine is unavailable: {_NATIVE_IMPORT_ERROR}"
    if requested == "cpp":
        raise NativeEngineUnavailable(message)
    warnings.warn(f"{message}; falling back to Python", RuntimeWarning, stacklevel=2)
    return MonteCarloDraft(context, player_list, manager_biases, seed), "python"


class NativeMonteCarloDraft:
    engine_name = "cpp"

    def __init__(
        self,
        context: LeagueContext,
        players: Iterable[Player],
        manager_biases: dict[str, dict[str, float]] | None = None,
        seed: int = 2026,
    ) -> None:
        if _NativeDraftEngine is None:
            raise NativeEngineUnavailable(
                f"native C++ engine is unavailable: {_NATIVE_IMPORT_ERROR}"
            )
        self.context = context
        self.manager_biases = manager_biases or {}
        self.seed = seed
        self._source_players = list(players)
        self._reference = MonteCarloDraft(
            context, self._source_players, self.manager_biases, seed
        )
        occupied_picks = {int(pick["pick_no"]) for pick in context.picks}
        if (
            not self._reference._standard_slots
            or context.rules.teams < 2
            or context.rules.rounds > 64
            or any(capacity > context.rules.rounds for capacity in self._reference._roster_capacities.values())
            or occupied_picks != set(range(1, len(occupied_picks) + 1))
        ):
            raise NativeEngineUnavailable(
                "this roster layout, traded-pick distribution, or future keeper board requires the Python engine"
            )
        self.players = self._reference.players
        self.by_id = self._reference.by_id
        self._player_indices = {
            player.sleeper_id: index for index, player in enumerate(self.players)
        }
        self._roster_ids = self._ordered_roster_ids()
        self._roster_indices = {
            roster_id: index for index, roster_id in enumerate(self._roster_ids)
        }
        self._engine = _NativeDraftEngine(self._build_config())

    def _ordered_roster_ids(self) -> list[int]:
        slot_to_roster = {
            int(slot): int(roster)
            for slot, roster in self.context.draft.get(
                "slot_to_roster_id", {}
            ).items()
        }
        roster_ids = [
            slot_to_roster.get(slot, slot)
            for slot in range(1, self.context.rules.teams + 1)
        ]
        if len(set(roster_ids)) != self.context.rules.teams:
            raise ValueError("draft slots do not map to unique roster IDs")
        return roster_ids

    def _encoded_players(
        self,
    ) -> list[tuple[int, float, float, float, float, float, float, float, float]]:
        encoded = []
        for player in self.players:
            try:
                position = POSITION_INDEX[player.position]
            except KeyError as exc:
                raise ValueError(
                    f"native engine does not support position {player.position!r}"
                ) from exc
            adp_mean = player.adp if player.adp is not None else player.ecr
            adp_sigma = (
                max(1.25, player.adp_uncertainty)
                if player.adp is not None
                else max(1.25, player.uncertainty * 1.1)
            )
            projected = self._reference._projected_points(player)
            outcome_cv = (
                player.outcome_cv
                if player.projected_points > 0
                else min(
                    0.50,
                    0.20
                    + player.uncertainty / max(20.0, player.ecr + 20.0),
                )
            )
            outcome_sigma = math.sqrt(math.log1p(outcome_cv**2))
            outcome_mu = (math.log(projected) if projected > 0 else -math.inf) - 0.5 * outcome_sigma**2
            encoded.append(
                (
                    position,
                    float(player.value_rank),
                    float(player.ecr),
                    float(adp_mean),
                    float(adp_sigma),
                    outcome_mu,
                    outcome_sigma,
                    float(
                        self._reference._tier_gap_rank.get(player.sleeper_id, 0.0)
                    ),
                    projected,
                )
            )
        return encoded

    def _state(self) -> tuple[list[list[int]], int]:
        rosters = []
        for roster_id in self._roster_ids:
            rosters.append(
                [
                    self._player_indices[player_id]
                    for player_id in self._reference.base_state.rosters.get(
                        roster_id, []
                    )
                ]
            )
        return rosters, self._reference.base_state.next_pick - 1

    def _build_config(self) -> dict[str, object]:
        rules = self.context.rules
        required = [rules.required_count(position) for position in POSITION_ORDER]
        caps = [
            rules.rounds if position in self._reference._eligible_positions else 0
            for position in POSITION_ORDER
        ]
        pick_owners = [
            self._roster_indices[roster_for_pick(self.context, pick,)]
            for pick in range(1, rules.teams * rules.rounds + 1)
        ]
        roster_to_user = self.context.roster_to_user
        biases = []
        for roster_id in self._roster_ids:
            user_id = roster_to_user.get(roster_id)
            position_biases = self.manager_biases.get(user_id or "", {})
            biases.append(
                [float(position_biases.get(position, 0.0)) for position in POSITION_ORDER]
            )
        rosters, next_pick = self._state()
        return {
            "teams": rules.teams,
            "rounds": rules.rounds,
            "flex_slots": rules.flex_slots,
            "required": required,
            "caps": caps,
            "user_roster": self._roster_indices[self.context.roster_id],
            "next_pick": next_pick,
            "pick_owners": pick_owners,
            "biases": biases,
            "has_projections": self._reference.has_projections,
            "pass_td": float(rules.scoring.get("pass_td", 4.0)),
            "players": self._encoded_players(),
            "value_order": [
                self._player_indices[player.sleeper_id]
                for player in self._reference.value_order
            ],
            "initial_rosters": rosters,
        }

    @staticmethod
    def _threads(workers: int | str) -> int:
        if workers == "auto":
            return 0
        return max(1, int(workers))

    def update_picks(self, picks: list[dict], seed: int) -> None:
        self.context.picks = picks
        self.seed = seed
        reference = MonteCarloDraft(
            self.context, self._source_players, self.manager_biases, seed
        )
        new_ids = [player.sleeper_id for player in reference.players]
        old_ids = [player.sleeper_id for player in self.players]
        self._reference = reference
        if new_ids != old_ids:
            self.players = reference.players
            self.by_id = reference.by_id
            self._player_indices = {
                player.sleeper_id: index for index, player in enumerate(self.players)
            }
            self._engine = _NativeDraftEngine(self._build_config())
            return
        rosters, next_pick = self._state()
        self._engine.set_state(rosters, next_pick)

    def recommend(
        self,
        simulations: int = 3000,
        candidate_count: int = 10,
        workers: int | str = 1,
    ) -> SimulationReport:
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        if candidate_count < 1:
            raise ValueError("candidate_count must be at least 1")
        next_user_pick = next_pick_for_roster(
            self.context,
            self._reference.base_state.next_pick,
            self.context.roster_id,
        )
        if next_user_pick > self.context.rules.teams * self.context.rules.rounds:
            return SimulationReport(next_user_pick, False, 0, ())
        on_clock = next_user_pick == self._reference.base_state.next_pick
        if on_clock:
            return self._recommend_on_clock(
                simulations, candidate_count, next_user_pick, workers
            )
        return self._recommend_before_turn(simulations, next_user_pick, workers)

    def _recommend_on_clock(
        self,
        simulations: int,
        candidate_count: int,
        next_user_pick: int,
        workers: int | str,
    ) -> SimulationReport:
        candidate_indices = self._engine.candidate_indices(candidate_count)
        if not candidate_indices:
            raise RuntimeError("No legal player is available for the user's next pick")
        raw = self._engine.recommend_on_clock(
            max(simulations, 25 * len(candidate_indices)), candidate_count, self.seed, self._threads(workers)
        )
        recommendations = []
        for player_index, score_sum, top_count, samples in zip(
            raw["candidate_indices"],
            raw["score_sums"],
            raw["top_counts"],
            raw["samples"],
            strict=True,
        ):
            player = self.players[player_index]
            starter_gain, next_pick_availability, wait_cost = (
                self._reference.recommendation_metrics(player)
            )
            recommendations.append(
                Recommendation(
                    player=player,
                    availability_rate=1.0,
                    selection_rate=1.0,
                    mean_score=score_sum / samples,
                    top_roster_rate=top_count / samples,
                    samples=samples,
                    starter_gain=starter_gain,
                    next_pick_availability=next_pick_availability,
                    wait_cost=wait_cost,
                )
            )
        recommendations.sort(
            key=lambda item: (item.mean_score, item.top_roster_rate), reverse=True
        )
        return SimulationReport(
            next_user_pick=next_user_pick,
            on_clock=True,
            total_rollouts=raw["total_rollouts"],
            recommendations=tuple(recommendations),
        )

    def _recommend_before_turn(
        self,
        simulations: int,
        next_user_pick: int,
        workers: int | str,
    ) -> SimulationReport:
        raw = self._engine.recommend_before_turn(
            simulations, self.seed, self._threads(workers)
        )
        selected = Counter(
            {
                player_index: count
                for player_index, count in enumerate(raw["selected_counts"])
                if count
            }
        )
        recommendations = []
        for player_index, samples in selected.most_common(12):
            player = self.players[player_index]
            starter_gain, next_pick_availability, wait_cost = (
                self._reference.recommendation_metrics(player)
            )
            recommendations.append(
                Recommendation(
                    player=player,
                    availability_rate=raw["available_counts"][player_index]
                    / simulations,
                    selection_rate=samples / simulations,
                    mean_score=raw["score_sums"][player_index] / samples,
                    top_roster_rate=raw["top_counts"][player_index] / samples,
                    samples=samples,
                    starter_gain=starter_gain,
                    next_pick_availability=next_pick_availability,
                    wait_cost=wait_cost,
                )
            )
        return SimulationReport(
            next_user_pick=next_user_pick,
            on_clock=False,
            total_rollouts=raw["total_rollouts"],
            recommendations=tuple(recommendations),
        )
