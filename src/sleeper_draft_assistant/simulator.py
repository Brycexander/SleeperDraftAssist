from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from collections import Counter, defaultdict
from dataclasses import dataclass
import math
import os
from typing import Iterable

import numpy as np

from .models import (
    DraftState,
    FLEX_POSITIONS,
    LeagueContext,
    LeagueRules,
    Player,
    Recommendation,
    normalize_position,
)


@dataclass(frozen=True, slots=True)
class SimulationReport:
    next_user_pick: int
    on_clock: bool
    total_rollouts: int
    recommendations: tuple[Recommendation, ...]


@dataclass(frozen=True, slots=True)
class RolloutResult:
    first_selection: str
    user_score: float
    top_roster: bool
    availability: frozenset[str]
    roster_scores: dict[int, float]
    user_picks: tuple[tuple[int, str], ...]


@dataclass(frozen=True, slots=True)
class PlayerFrequency:
    player: Player
    roster_rate: float
    average_round: float


@dataclass(frozen=True, slots=True)
class RepresentativeRun:
    finish: int
    wins: float
    model_score: float
    picks: tuple[tuple[int, Player], ...]


@dataclass(frozen=True, slots=True)
class DraftAnalysisReport:
    simulations: int
    regular_season_weeks: int
    playoff_teams: int
    playoff_rate: float
    top_two_rate: float
    average_finish: float
    average_wins: float
    finish_rates: tuple[float, ...]
    common_players: tuple[PlayerFrequency, ...]
    common_openings: tuple[tuple[tuple[str, ...], float], ...]
    common_builds: tuple[tuple[tuple[tuple[str, int], ...], float], ...]
    representative_run: RepresentativeRun


@dataclass(frozen=True, slots=True)
class _SeasonResult:
    finish: int
    wins: float
    points: float


def _resolve_workers(workers: int | str | None) -> int:
    if workers in (None, 1, "1"):
        return 1
    if workers == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    return max(1, int(workers))


def _simulation_chunks(simulations: int, workers: int) -> list[int]:
    workers = min(workers, simulations)
    base, extra = divmod(simulations, workers)
    return [base + int(index < extra) for index in range(workers)]


def _recommend_before_turn_worker(
    context: LeagueContext,
    players: list[Player],
    manager_biases: dict[str, dict[str, float]],
    seed: int,
    simulations: int,
) -> tuple[Counter[str], Counter[str], dict[str, float], Counter[str]]:
    simulator = MonteCarloDraft(context, players, manager_biases, seed=seed)
    selected = Counter[str]()
    available = Counter[str]()
    score_sum = defaultdict(float)
    top_sum = Counter[str]()

    for _ in range(simulations):
        order, ranks = simulator._sample_opponent_board()
        result = simulator._finish_rollout(simulator.base_state.clone(), order, ranks)
        selected[result.first_selection] += 1
        score_sum[result.first_selection] += result.user_score
        top_sum[result.first_selection] += int(result.top_roster)
        available.update(result.availability)

    return selected, available, dict(score_sum), top_sum


def _recommend_on_clock_worker(
    context: LeagueContext,
    players: list[Player],
    manager_biases: dict[str, dict[str, float]],
    seed: int,
    candidate_id: str,
    simulations: int,
) -> tuple[str, float, int, int]:
    simulator = MonteCarloDraft(context, players, manager_biases, seed=seed)
    candidate = simulator.by_id[candidate_id]
    score_sum = 0.0
    top_count = 0

    for _ in range(simulations):
        order, ranks = simulator._sample_opponent_board()
        result = simulator._finish_rollout(
            simulator.base_state.clone(), order, ranks, forced_first_pick=candidate
        )
        score_sum += result.user_score
        top_count += int(result.top_roster)

    return candidate_id, score_sum, top_count, simulations


def _analyze_worker(
    context: LeagueContext,
    players: list[Player],
    manager_biases: dict[str, dict[str, float]],
    seed: int,
    simulations: int,
    weekly_variance: float,
) -> tuple[
    Counter[str],
    dict[str, float],
    Counter[tuple[str, ...]],
    Counter[tuple[tuple[str, int], ...]],
    Counter[int],
    int,
    int,
    float,
    float,
    list[tuple[_SeasonResult, RolloutResult]],
]:
    simulator = MonteCarloDraft(context, players, manager_biases, seed=seed)
    settings = context.league.get("settings") or {}
    playoff_teams = min(
        context.rules.teams,
        int(settings.get("playoff_teams", min(6, context.rules.teams))),
    )
    player_counts = Counter[str]()
    player_rounds = defaultdict(float)
    opening_counts = Counter[tuple[str, ...]]()
    build_counts = Counter[tuple[tuple[str, int], ...]]()
    finish_counts = Counter[int]()
    playoff_count = 0
    top_two_count = 0
    wins_total = 0.0
    finishes_total = 0.0
    run_data: list[tuple[_SeasonResult, RolloutResult]] = []

    for _ in range(simulations):
        order, ranks = simulator._sample_opponent_board()
        rollout = simulator._finish_rollout(simulator.base_state.clone(), order, ranks)
        season = simulator._simulate_regular_season(rollout.roster_scores, weekly_variance)
        run_data.append((season, rollout))
        finish_counts[season.finish] += 1
        playoff_count += int(season.finish <= playoff_teams)
        top_two_count += int(season.finish <= min(2, playoff_teams))
        wins_total += season.wins
        finishes_total += season.finish

        for round_number, player_id in rollout.user_picks:
            player_counts[player_id] += 1
            player_rounds[player_id] += round_number
        opening = tuple(
            simulator.by_id[player_id].position
            for _, player_id in rollout.user_picks[: min(4, len(rollout.user_picks))]
        )
        opening_counts[opening] += 1
        build = tuple(sorted(simulator._position_counts([p for _, p in rollout.user_picks]).items()))
        build_counts[build] += 1

    return (
        player_counts,
        dict(player_rounds),
        opening_counts,
        build_counts,
        finish_counts,
        playoff_count,
        top_two_count,
        wins_total,
        finishes_total,
        run_data,
    )


def snake_slot_for_pick(pick_number: int, teams: int) -> int:
    round_index, index_in_round = divmod(pick_number - 1, teams)
    if round_index % 2 == 0:
        return index_in_round + 1
    return teams - index_in_round


def roster_for_pick(context: LeagueContext, pick_number: int) -> int:
    slot = snake_slot_for_pick(pick_number, context.rules.teams)
    slot_to_roster = {
        int(key): int(value)
        for key, value in context.draft.get("slot_to_roster_id", {}).items()
    }
    original_roster = slot_to_roster.get(slot, slot)
    round_number = (pick_number - 1) // context.rules.teams + 1
    for traded in context.traded_picks:
        if (
            int(traded.get("round", -1)) == round_number
            and int(traded.get("roster_id", -1)) == original_roster
        ):
            return int(traded["owner_id"])
    return original_roster


def next_pick_for_roster(context: LeagueContext, start_pick: int, roster_id: int) -> int:
    final_pick = context.rules.teams * context.rules.rounds
    for pick_number in range(start_pick, final_pick + 1):
        if roster_for_pick(context, pick_number) == roster_id:
            return pick_number
    return final_pick + 1


class MonteCarloDraft:
    def __init__(
        self,
        context: LeagueContext,
        players: Iterable[Player],
        manager_biases: dict[str, dict[str, float]] | None = None,
        seed: int = 2026,
    ) -> None:
        self.context = context
        self.rules = context.rules
        self.players = list(players)
        self.by_id = {player.sleeper_id: player for player in self.players}
        self.manager_biases = manager_biases or {}
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self._add_missing_picked_players()
        self.has_projections = any(player.projected_points > 0 for player in self.players)
        self.value_order = sorted(
            self.players,
            key=lambda player: player.value_rank if self.has_projections else player.ecr,
        )
        self.base_state = self._state_from_picks()

    def _add_missing_picked_players(self) -> None:
        for pick in self.context.picks:
            player_id = str(pick["player_id"])
            if player_id in self.by_id:
                continue
            metadata = pick.get("metadata") or {}
            player = Player(
                sleeper_id=player_id,
                name=" ".join(
                    part for part in (metadata.get("first_name"), metadata.get("last_name")) if part
                )
                or metadata.get("team")
                or player_id,
                position=normalize_position(metadata.get("position")),
                team=str(metadata.get("team") or "FA"),
                ecr=400.0,
                uncertainty=25.0,
            )
            self.players.append(player)
            self.by_id[player_id] = player

    def _state_from_picks(self) -> DraftState:
        roster_ids = set(self.context.roster_to_user)
        roster_ids.update(range(1, self.rules.teams + 1))
        rosters = {roster_id: [] for roster_id in roster_ids}
        drafted: set[str] = set()
        highest_pick = 0
        for pick in sorted(self.context.picks, key=lambda item: int(item["pick_no"])):
            player_id = str(pick["player_id"])
            roster_id = int(pick["roster_id"])
            rosters.setdefault(roster_id, []).append(player_id)
            drafted.add(player_id)
            highest_pick = max(highest_pick, int(pick["pick_no"]))
        return DraftState(next_pick=highest_pick + 1, rosters=rosters, drafted=drafted)

    def _position_counts(self, roster: list[str]) -> Counter[str]:
        return Counter(
            self.by_id[player_id].position
            for player_id in roster
            if player_id in self.by_id
        )

    def _missing_mandatory(self, counts: Counter[str]) -> Counter[str]:
        missing: Counter[str] = Counter()
        for position in ("QB", "RB", "WR", "TE", "K", "DEF"):
            needed = self.rules.required_count(position) - counts[position]
            if needed > 0:
                missing[position] = needed
        return missing

    def _can_add(
        self,
        player: Player,
        roster_size: int,
        counts: Counter[str],
        missing: Counter[str],
    ) -> bool:
        if roster_size >= self.rules.rounds:
            return False
        if player.position in {"K", "DEF"} and roster_size < self.rules.rounds - 3:
            return False
        skill_cap = max(4, self.rules.flex_slots + 4)
        caps = {
            "QB": max(1, self.rules.required_count("QB")),
            "RB": max(skill_cap, self.rules.required_count("RB")),
            "WR": max(skill_cap, self.rules.required_count("WR")),
            "TE": max(2, self.rules.required_count("TE")),
            "K": max(1, self.rules.required_count("K")),
            "DEF": max(1, self.rules.required_count("DEF")),
        }
        if counts[player.position] >= caps.get(player.position, self.rules.rounds):
            return False

        remaining_after_pick = self.rules.rounds - roster_size - 1
        missing_after_pick = sum(missing.values()) - int(missing[player.position] > 0)
        return missing_after_pick <= remaining_after_pick

    def _pick_score(
        self,
        player: Player,
        counts: Counter[str],
        round_number: int,
        rank: float,
        manager_user_id: str | None = None,
    ) -> float:
        score = float(rank)
        required = self.rules.required_count(player.position)

        if counts[player.position] < required:
            score -= 4.0
        if player.position in {"RB", "WR"}:
            flex_target = required + math.ceil(self.rules.flex_slots / 2)
            if counts[player.position] < flex_target:
                score -= 2.0
            other_position = "WR" if player.position == "RB" else "RB"
            imbalance = counts[other_position] - counts[player.position]
            score -= max(0, imbalance - 1) * 10.0
        elif player.position == "TE" and counts["TE"] >= max(1, required):
            score += 15.0
        elif player.position == "QB":
            if not self.has_projections:
                score += max(0, 12 - self.rules.teams) * 2.0
                score -= max(0.0, self.rules.scoring.get("pass_td", 4.0) - 4.0) * 3.0
            if counts["QB"] >= 1:
                score += 30.0 if round_number < 12 else 18.0
        elif player.position in {"K", "DEF"} and round_number < self.rules.rounds - 2:
            score += 85.0

        if manager_user_id:
            score += self.manager_biases.get(manager_user_id, {}).get(player.position, 0.0)
        return score

    def _ordered_candidates(
        self,
        state: DraftState,
        roster_id: int,
        order: list[Player],
        ranks: dict[str, float] | None,
        manager_user_id: str | None,
        limit: int = 36,
    ) -> list[Player]:
        roster = state.rosters.setdefault(roster_id, [])
        counts = self._position_counts(roster)
        missing = self._missing_mandatory(counts)
        round_number = (state.next_pick - 1) // self.rules.teams + 1
        candidates: list[tuple[float, Player]] = []
        for player in order:
            if player.sleeper_id in state.drafted or not self._can_add(
                player, len(roster), counts, missing
            ):
                continue
            rank = (
                ranks[player.sleeper_id]
                if ranks is not None
                else (player.value_rank if self.has_projections else player.ecr)
            )
            score = self._pick_score(player, counts, round_number, rank, manager_user_id)
            candidates.append((score, player))
            if len(candidates) >= limit:
                break
        candidates.sort(key=lambda item: item[0])
        return [player for _, player in candidates]

    def _select(
        self,
        state: DraftState,
        roster_id: int,
        order: list[Player],
        ranks: dict[str, float] | None,
        manager_user_id: str | None,
    ) -> Player:
        candidates = self._ordered_candidates(
            state, roster_id, order, ranks, manager_user_id, limit=40
        )
        if not candidates:
            raise RuntimeError(f"No legal player available for roster {roster_id}")
        return candidates[0]

    @staticmethod
    def _draft_player(state: DraftState, roster_id: int, player: Player) -> None:
        state.rosters.setdefault(roster_id, []).append(player.sleeper_id)
        state.drafted.add(player.sleeper_id)
        state.next_pick += 1

    def _sample_opponent_board(self) -> tuple[list[Player], dict[str, float]]:
        sampled = self.rng.normal(
            [player.adp if player.adp is not None else player.ecr for player in self.players],
            [
                max(1.25, player.adp_uncertainty)
                if player.adp is not None
                else max(1.25, player.uncertainty * 1.1)
                for player in self.players
            ],
        )
        sampled = np.clip(sampled, 1.0, 500.0)
        ranks = {
            player.sleeper_id: float(rank)
            for player, rank in zip(self.players, sampled, strict=True)
        }
        return sorted(self.players, key=lambda player: ranks[player.sleeper_id]), ranks

    @staticmethod
    def _fallback_projected_points(player: Player) -> float:
        return 17.0 * 220.0 / math.log2(player.ecr + 4.0)

    def _sample_outcome_points(self) -> dict[str, float]:
        means = np.array(
            [
                player.projected_points
                if player.projected_points > 0
                else self._fallback_projected_points(player)
                for player in self.players
            ],
            dtype=float,
        )
        cvs = np.array(
            [
                player.outcome_cv
                if player.projected_points > 0
                else min(0.50, 0.20 + player.uncertainty / max(20.0, player.ecr + 20.0))
                for player in self.players
            ],
            dtype=float,
        )
        sigmas = np.sqrt(np.log1p(cvs**2))
        sampled = self.rng.lognormal(np.log(np.maximum(means, 1.0)) - 0.5 * sigmas**2, sigmas)
        return {
            player.sleeper_id: float(points)
            for player, points in zip(self.players, sampled, strict=True)
        }

    def _lineup_score(self, roster: list[str], outcome_points: dict[str, float]) -> float:
        remaining = [self.by_id[player_id] for player_id in roster if player_id in self.by_id]
        selected: list[Player] = []

        def take(position: str, count: int) -> None:
            eligible = [player for player in remaining if player.position == position]
            eligible.sort(
                key=lambda player: outcome_points[player.sleeper_id],
                reverse=True,
            )
            for player in eligible[:count]:
                selected.append(player)
                remaining.remove(player)

        for position in ("QB", "RB", "WR", "TE", "K", "DEF"):
            take(position, self.rules.required_count(position))

        flex = [player for player in remaining if player.position in FLEX_POSITIONS]
        flex.sort(
            key=lambda player: outcome_points[player.sleeper_id],
            reverse=True,
        )
        for player in flex[: self.rules.flex_slots]:
            selected.append(player)
            remaining.remove(player)

        starter_score = sum(outcome_points[player.sleeper_id] for player in selected)
        bench = sorted(
            (
                outcome_points[player.sleeper_id]
                for player in remaining
                if player.position in FLEX_POSITIONS | {"QB"}
            ),
            reverse=True,
        )
        return (starter_score + 0.08 * sum(bench[:4])) / 17.0

    def _finish_rollout(
        self,
        state: DraftState,
        opponent_order: list[Player],
        opponent_ranks: dict[str, float],
        forced_first_pick: Player | None = None,
    ) -> RolloutResult:
        final_pick = self.rules.teams * self.rules.rounds
        first_user_pick = next_pick_for_roster(
            self.context, state.next_pick, self.context.roster_id
        )
        first_selection = ""
        availability: set[str] = set()
        user_picks = [
            (int(pick["round"]), str(pick["player_id"]))
            for pick in sorted(self.context.picks, key=lambda item: int(item["pick_no"]))
            if int(pick["roster_id"]) == self.context.roster_id
        ]

        while state.next_pick <= final_pick:
            pick_number = state.next_pick
            round_number = (pick_number - 1) // self.rules.teams + 1
            roster_id = roster_for_pick(self.context, pick_number)
            user_id = self.context.roster_to_user.get(roster_id)

            if pick_number == first_user_pick:
                availability = {
                    player.sleeper_id
                    for player in self.value_order[:40]
                    if player.sleeper_id not in state.drafted
                }

            if roster_id == self.context.roster_id:
                if pick_number == first_user_pick and forced_first_pick is not None:
                    if forced_first_pick.sleeper_id in state.drafted:
                        raise RuntimeError("Forced candidate was drafted before the user's pick")
                    player = forced_first_pick
                else:
                    player = self._select(
                        state, roster_id, self.value_order, None, None
                    )
                if not first_selection:
                    first_selection = player.sleeper_id
                user_picks.append((round_number, player.sleeper_id))
            else:
                player = self._select(
                    state, roster_id, opponent_order, opponent_ranks, user_id
                )
            self._draft_player(state, roster_id, player)

        outcome_points = self._sample_outcome_points()
        scores = {
            roster_id: self._lineup_score(roster, outcome_points)
            for roster_id, roster in state.rosters.items()
        }
        user_score = scores[self.context.roster_id]
        return RolloutResult(
            first_selection=first_selection,
            user_score=user_score,
            top_roster=user_score >= max(scores.values()),
            availability=frozenset(availability),
            roster_scores=scores,
            user_picks=tuple(user_picks),
        )

    @staticmethod
    def _round_robin(roster_ids: list[int]) -> list[list[tuple[int, int]]]:
        teams: list[int | None] = list(roster_ids)
        if len(teams) % 2:
            teams.append(None)
        rounds: list[list[tuple[int, int]]] = []
        for _ in range(len(teams) - 1):
            pairs = []
            for index in range(len(teams) // 2):
                home = teams[index]
                away = teams[-index - 1]
                if home is not None and away is not None:
                    pairs.append((home, away))
            rounds.append(pairs)
            teams = [teams[0], teams[-1], *teams[1:-1]]
        return rounds

    def _simulate_regular_season(
        self, roster_scores: dict[int, float], weekly_variance: float
    ) -> _SeasonResult:
        settings = self.context.league.get("settings") or {}
        playoff_week_start = int(settings.get("playoff_week_start", 15))
        start_week = int(settings.get("start_week", 1))
        weeks = max(1, playoff_week_start - start_week)
        roster_ids = sorted(roster_scores)
        schedule = self._round_robin(roster_ids)
        wins = {roster_id: 0.0 for roster_id in roster_ids}
        points = {roster_id: 0.0 for roster_id in roster_ids}

        for week in range(weeks):
            weekly_scores = {
                roster_id: roster_scores[roster_id]
                * float(
                    self.rng.lognormal(
                        mean=-0.5 * weekly_variance**2,
                        sigma=weekly_variance,
                    )
                )
                for roster_id in roster_ids
            }
            for roster_id, score in weekly_scores.items():
                points[roster_id] += score
            for home, away in schedule[week % len(schedule)]:
                if weekly_scores[home] > weekly_scores[away]:
                    wins[home] += 1.0
                elif weekly_scores[away] > weekly_scores[home]:
                    wins[away] += 1.0
                else:
                    wins[home] += 0.5
                    wins[away] += 0.5

        standings = sorted(
            roster_ids,
            key=lambda roster_id: (wins[roster_id], points[roster_id]),
            reverse=True,
        )
        return _SeasonResult(
            finish=standings.index(self.context.roster_id) + 1,
            wins=wins[self.context.roster_id],
            points=points[self.context.roster_id],
        )

    def analyze(
        self,
        simulations: int = 1000,
        weekly_variance: float = 0.22,
        top_players: int = 15,
        workers: int | str = 1,
    ) -> DraftAnalysisReport:
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        if not 0.0 <= weekly_variance <= 1.0:
            raise ValueError("weekly_variance must be between 0 and 1")

        settings = self.context.league.get("settings") or {}
        playoff_teams = min(
            self.rules.teams,
            int(settings.get("playoff_teams", min(6, self.rules.teams))),
        )
        playoff_week_start = int(settings.get("playoff_week_start", 15))
        start_week = int(settings.get("start_week", 1))
        weeks = max(1, playoff_week_start - start_week)
        player_counts = Counter[str]()
        player_rounds = defaultdict(float)
        opening_counts = Counter[tuple[str, ...]]()
        build_counts = Counter[tuple[tuple[str, int], ...]]()
        finish_counts = Counter[int]()
        playoff_count = 0
        top_two_count = 0
        wins_total = 0.0
        finishes_total = 0.0
        run_data: list[tuple[_SeasonResult, RolloutResult]] = []

        worker_count = _resolve_workers(workers)
        if worker_count == 1:
            for _ in range(simulations):
                order, ranks = self._sample_opponent_board()
                rollout = self._finish_rollout(self.base_state.clone(), order, ranks)
                season = self._simulate_regular_season(rollout.roster_scores, weekly_variance)
                run_data.append((season, rollout))
                finish_counts[season.finish] += 1
                playoff_count += int(season.finish <= playoff_teams)
                top_two_count += int(season.finish <= min(2, playoff_teams))
                wins_total += season.wins
                finishes_total += season.finish

                for round_number, player_id in rollout.user_picks:
                    player_counts[player_id] += 1
                    player_rounds[player_id] += round_number
                opening = tuple(
                    self.by_id[player_id].position
                    for _, player_id in rollout.user_picks[: min(4, len(rollout.user_picks))]
                )
                opening_counts[opening] += 1
                build = tuple(sorted(self._position_counts([p for _, p in rollout.user_picks]).items()))
                build_counts[build] += 1
        else:
            chunks = _simulation_chunks(simulations, worker_count)
            seeds = [self.seed + 100_000 + index for index in range(len(chunks))]
            with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
                worker_results = list(
                    executor.map(
                        _analyze_worker,
                        [self.context] * len(chunks),
                        [self.players] * len(chunks),
                        [self.manager_biases] * len(chunks),
                        seeds,
                        chunks,
                        [weekly_variance] * len(chunks),
                    )
                )

            for (
                chunk_player_counts,
                chunk_player_rounds,
                chunk_opening_counts,
                chunk_build_counts,
                chunk_finish_counts,
                chunk_playoff_count,
                chunk_top_two_count,
                chunk_wins_total,
                chunk_finishes_total,
                chunk_run_data,
            ) in worker_results:
                player_counts.update(chunk_player_counts)
                for player_id, total_round in chunk_player_rounds.items():
                    player_rounds[player_id] += total_round
                opening_counts.update(chunk_opening_counts)
                build_counts.update(chunk_build_counts)
                finish_counts.update(chunk_finish_counts)
                playoff_count += chunk_playoff_count
                top_two_count += chunk_top_two_count
                wins_total += chunk_wins_total
                finishes_total += chunk_finishes_total
                run_data.extend(chunk_run_data)

        offensive_player_counts = [
            (player_id, count)
            for player_id, count in player_counts.most_common()
            if self.by_id[player_id].position not in {"K", "DEF"}
        ]
        common_players = tuple(
            PlayerFrequency(
                player=self.by_id[player_id],
                roster_rate=count / simulations,
                average_round=player_rounds[player_id] / count,
            )
            for player_id, count in offensive_player_counts[:top_players]
        )
        common_openings = tuple(
            (opening, count / simulations)
            for opening, count in opening_counts.most_common(5)
        )
        common_builds = tuple(
            (build, count / simulations) for build, count in build_counts.most_common(5)
        )

        median_finish = float(np.median([season.finish for season, _ in run_data]))
        median_wins = float(np.median([season.wins for season, _ in run_data]))
        median_score = float(np.median([rollout.user_score for _, rollout in run_data]))
        score_scale = max(
            1.0, float(np.std([rollout.user_score for _, rollout in run_data]))
        )
        representative_season, representative_rollout = min(
            run_data,
            key=lambda item: (
                abs(item[0].finish - median_finish) / self.rules.teams
                + abs(item[0].wins - median_wins) / weeks
                + abs(item[1].user_score - median_score) / score_scale
            ),
        )
        representative = RepresentativeRun(
            finish=representative_season.finish,
            wins=representative_season.wins,
            model_score=representative_rollout.user_score,
            picks=tuple(
                (round_number, self.by_id[player_id])
                for round_number, player_id in representative_rollout.user_picks
            ),
        )
        return DraftAnalysisReport(
            simulations=simulations,
            regular_season_weeks=weeks,
            playoff_teams=playoff_teams,
            playoff_rate=playoff_count / simulations,
            top_two_rate=top_two_count / simulations,
            average_finish=finishes_total / simulations,
            average_wins=wins_total / simulations,
            finish_rates=tuple(
                finish_counts[finish] / simulations
                for finish in range(1, self.rules.teams + 1)
            ),
            common_players=common_players,
            common_openings=common_openings,
            common_builds=common_builds,
            representative_run=representative,
        )

    def recommend(
        self,
        simulations: int = 3000,
        candidate_count: int = 10,
        workers: int | str = 1,
    ) -> SimulationReport:
        if simulations < 1:
            raise ValueError("simulations must be at least 1")
        next_user_pick = next_pick_for_roster(
            self.context, self.base_state.next_pick, self.context.roster_id
        )
        on_clock = next_user_pick == self.base_state.next_pick
        if on_clock:
            return self._recommend_on_clock(
                simulations, candidate_count, next_user_pick, workers
            )
        return self._recommend_before_turn(simulations, next_user_pick, workers)

    def _recommend_before_turn(
        self, simulations: int, next_user_pick: int, workers: int | str
    ) -> SimulationReport:
        selected = Counter[str]()
        available = Counter[str]()
        score_sum = defaultdict(float)
        top_sum = Counter[str]()

        worker_count = _resolve_workers(workers)
        if worker_count == 1:
            for _ in range(simulations):
                order, ranks = self._sample_opponent_board()
                result = self._finish_rollout(
                    self.base_state.clone(), order, ranks
                )
                selected[result.first_selection] += 1
                score_sum[result.first_selection] += result.user_score
                top_sum[result.first_selection] += int(result.top_roster)
                available.update(result.availability)
        else:
            chunks = _simulation_chunks(simulations, worker_count)
            seeds = [self.seed + 10_000 + index for index in range(len(chunks))]
            with ProcessPoolExecutor(max_workers=len(chunks)) as executor:
                worker_results = list(
                    executor.map(
                        _recommend_before_turn_worker,
                        [self.context] * len(chunks),
                        [self.players] * len(chunks),
                        [self.manager_biases] * len(chunks),
                        seeds,
                        chunks,
                    )
                )

            for chunk_selected, chunk_available, chunk_score_sum, chunk_top_sum in worker_results:
                selected.update(chunk_selected)
                available.update(chunk_available)
                top_sum.update(chunk_top_sum)
                for player_id, score in chunk_score_sum.items():
                    score_sum[player_id] += score

        recommendations = []
        for player_id, samples in selected.most_common(12):
            player = self.by_id[player_id]
            recommendations.append(
                Recommendation(
                    player=player,
                    availability_rate=available[player_id] / simulations,
                    selection_rate=samples / simulations,
                    mean_score=score_sum[player_id] / samples,
                    top_roster_rate=top_sum[player_id] / samples,
                    samples=samples,
                )
            )
        return SimulationReport(
            next_user_pick=next_user_pick,
            on_clock=False,
            total_rollouts=simulations,
            recommendations=tuple(recommendations),
        )

    def _recommend_on_clock(
        self,
        simulations: int,
        candidate_count: int,
        next_user_pick: int,
        workers: int | str,
    ) -> SimulationReport:
        candidates = self._ordered_candidates(
            self.base_state,
            self.context.roster_id,
            self.value_order,
            None,
            None,
            limit=max(40, candidate_count * 4),
        )[:candidate_count]
        per_candidate = max(25, simulations // max(1, len(candidates)))
        recommendations = []

        worker_count = _resolve_workers(workers)
        if worker_count == 1:
            for candidate in candidates:
                score_sum = 0.0
                top_count = 0
                for _ in range(per_candidate):
                    order, ranks = self._sample_opponent_board()
                    result = self._finish_rollout(
                        self.base_state.clone(), order, ranks, forced_first_pick=candidate
                    )
                    score_sum += result.user_score
                    top_count += int(result.top_roster)
                recommendations.append(
                    Recommendation(
                        player=candidate,
                        availability_rate=1.0,
                        selection_rate=1.0,
                        mean_score=score_sum / per_candidate,
                        top_roster_rate=top_count / per_candidate,
                        samples=per_candidate,
                    )
                )
        else:
            tasks = []
            for candidate_index, candidate in enumerate(candidates):
                chunks = _simulation_chunks(per_candidate, worker_count)
                for chunk_index, chunk in enumerate(chunks):
                    tasks.append(
                        (
                            self.seed + 20_000 + candidate_index * 1_000 + chunk_index,
                            candidate.sleeper_id,
                            chunk,
                        )
                    )
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                worker_results = list(
                    executor.map(
                        _recommend_on_clock_worker,
                        [self.context] * len(tasks),
                        [self.players] * len(tasks),
                        [self.manager_biases] * len(tasks),
                        [seed for seed, _, _ in tasks],
                        [candidate_id for _, candidate_id, _ in tasks],
                        [chunk for _, _, chunk in tasks],
                    )
                )

            by_candidate: dict[str, list[float | int]] = {
                candidate.sleeper_id: [0.0, 0, 0] for candidate in candidates
            }
            for candidate_id, score_sum, top_count, samples in worker_results:
                totals = by_candidate[candidate_id]
                totals[0] = float(totals[0]) + score_sum
                totals[1] = int(totals[1]) + top_count
                totals[2] = int(totals[2]) + samples

            for candidate in candidates:
                score_sum, top_count, samples = by_candidate[candidate.sleeper_id]
                recommendations.append(
                    Recommendation(
                        player=candidate,
                        availability_rate=1.0,
                        selection_rate=1.0,
                        mean_score=float(score_sum) / int(samples),
                        top_roster_rate=int(top_count) / int(samples),
                        samples=int(samples),
                    )
                )

        recommendations.sort(
            key=lambda item: (item.mean_score, item.top_roster_rate), reverse=True
        )
        return SimulationReport(
            next_user_pick=next_user_pick,
            on_clock=True,
            total_rollouts=per_candidate * len(candidates),
            recommendations=tuple(recommendations),
        )
