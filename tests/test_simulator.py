from __future__ import annotations

from dataclasses import replace
from collections import Counter

import pytest

from sleeper_draft_assistant.models import LeagueContext, LeagueRules, Player
from sleeper_draft_assistant.native_engine import (
    NativeEngineUnavailable,
    NativeMonteCarloDraft,
    create_recommendation_engine,
    native_available,
)
from sleeper_draft_assistant.simulator import (
    MonteCarloDraft,
    roster_for_pick,
    snake_slot_for_pick,
)
from sleeper_draft_assistant.draft_slots import assign_starters


def make_context(
    *,
    teams: int = 4,
    rounds: int = 4,
    positions: tuple[str, ...] = ("QB", "RB", "WR", "FLEX"),
) -> LeagueContext:
    draft_order = {f"user-{slot}": slot for slot in range(1, teams + 1)}
    slot_to_roster = {str(slot): slot for slot in range(1, teams + 1)}
    rules = LeagueRules(
        teams=teams,
        rounds=rounds,
        roster_positions=positions,
        scoring={"rec": 1.0, "pass_td": 6.0},
    )
    return LeagueContext(
        league_id="league",
        username="user-1",
        user_id="user-1",
        roster_id=1,
        draft_slot=1,
        league={
            "name": "Test",
            "season": "2026",
            "previous_league_id": None,
            "settings": {
                "playoff_teams": 2,
                "start_week": 1,
                "playoff_week_start": 5,
            },
        },
        draft={
            "draft_id": "draft",
            "draft_order": draft_order,
            "slot_to_roster_id": slot_to_roster,
            "settings": {"teams": teams, "rounds": rounds},
        },
        picks=[],
        traded_picks=[],
        users=[],
        rosters=[],
        rules=rules,
    )


def make_players(count_per_position: int = 8) -> list[Player]:
    players = []
    overall_rank = 1
    for position in ("RB", "WR", "QB", "TE"):
        for index in range(count_per_position):
            players.append(
                Player(
                    sleeper_id=f"{position}-{index}",
                    name=f"{position} Player {index}",
                    position=position,
                    team="TST",
                    ecr=float(overall_rank),
                    uncertainty=2.0,
                )
            )
            overall_rank += 1
    return players


def test_snake_slots_reverse_in_even_rounds() -> None:
    assert [snake_slot_for_pick(pick, 4) for pick in range(1, 9)] == [
        1,
        2,
        3,
        4,
        4,
        3,
        2,
        1,
    ]


def test_traded_pick_changes_roster_owner() -> None:
    context = make_context()
    context.traded_picks = [{"round": 2, "roster_id": 4, "owner_id": 2}]
    assert roster_for_pick(context, 5) == 2


def test_recommendation_runs_from_an_empty_board() -> None:
    simulator = MonteCarloDraft(make_context(), make_players(), seed=7)
    report = simulator.recommend(simulations=60, candidate_count=3)

    assert report.on_clock
    assert report.next_user_pick == 1
    assert len(report.recommendations) == 3
    assert report.total_rollouts >= 75
    assert all(item.samples >= 25 for item in report.recommendations)


def test_parallel_recommendation_accounts_for_rollouts() -> None:
    simulator = MonteCarloDraft(make_context(), make_players(), seed=7)
    report = simulator.recommend(simulations=80, candidate_count=2, workers=2)

    assert report.on_clock
    assert len(report.recommendations) == 2
    assert report.total_rollouts == 80
    assert sum(item.samples for item in report.recommendations) == 80


def test_existing_picks_advance_the_board() -> None:
    context = make_context()
    context.picks = [
        {"pick_no": 1, "roster_id": 1, "player_id": "RB-0", "metadata": {"position": "RB"}},
        {"pick_no": 2, "roster_id": 2, "player_id": "RB-1", "metadata": {"position": "RB"}},
    ]
    simulator = MonteCarloDraft(context, make_players(), seed=9)

    assert simulator.base_state.next_pick == 3
    assert simulator.base_state.rosters[1] == ["RB-0"]
    assert simulator.base_state.drafted == {"RB-0", "RB-1"}


def test_recommendation_metrics_separate_lineup_gain_and_wait_cost() -> None:
    players = []
    for index, player in enumerate(make_players()):
        players.append(
            replace(
                player,
                projected_points=max(20.0, 200.0 - index * 3.0),
                vorp=max(0.0, 100.0 - index * 4.0),
                value_rank=float(index + 1),
                adp=float(index + 2),
                adp_uncertainty=1.5,
            )
        )
    simulator = MonteCarloDraft(make_context(), players, seed=7)

    starter_gain, next_availability, wait_cost = simulator.recommendation_metrics(
        players[0]
    )

    assert starter_gain == pytest.approx(200.0)
    assert next_availability is not None
    assert next_availability < 0.01
    assert wait_cost > 0


def test_analysis_summarizes_drafts_and_playoff_results() -> None:
    simulator = MonteCarloDraft(make_context(), make_players(), seed=17)
    report = simulator.analyze(simulations=40, weekly_variance=0.14, top_players=6)

    assert report.simulations == 40
    assert report.regular_season_weeks == 4
    assert report.playoff_teams == 2
    assert 0.0 <= report.playoff_rate <= 1.0
    assert 0.0 <= report.top_two_rate <= 1.0
    assert sum(report.finish_rates) == 1.0
    assert len(report.finish_rates) == 4
    assert len(report.common_players) == 6
    assert len(report.representative_run.picks) == 4
    assert 0.0 <= report.average_wins <= report.regular_season_weeks


def test_parallel_analysis_summarizes_requested_simulations() -> None:
    simulator = MonteCarloDraft(make_context(), make_players(), seed=17)
    report = simulator.analyze(
        simulations=40,
        weekly_variance=0.14,
        top_players=6,
        workers=2,
    )

    assert report.simulations == 40
    assert sum(report.finish_rates) == 1.0
    assert len(report.common_players) == 6


def test_round_robin_pairs_every_team_once_per_round() -> None:
    rounds = MonteCarloDraft._round_robin([1, 2, 3, 4])

    assert len(rounds) == 3
    assert all(sorted(team for pair in week for team in pair) == [1, 2, 3, 4] for week in rounds)
    assert {tuple(sorted(pair)) for week in rounds for pair in week} == {
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    }


@pytest.mark.skipif(not native_available(), reason="native extension is not built")
def test_native_candidate_pool_matches_python_policy() -> None:
    context = make_context()
    players = make_players()
    python = MonteCarloDraft(context, players, seed=7)
    native = NativeMonteCarloDraft(context, players, seed=7)

    python_candidates = python._ordered_candidates(
        python.base_state,
        context.roster_id,
        python.value_order,
        None,
        None,
        limit=40,
    )[:6]
    native_indices = native._engine.candidate_indices(6)

    assert [player.sleeper_id for player in python_candidates] == [
        native.players[index].sleeper_id for index in native_indices
    ]


@pytest.mark.skipif(not native_available(), reason="native extension is not built")
def test_native_candidate_pool_uses_ecr_without_projections() -> None:
    context = make_context()
    players = [
        replace(player, value_rank=1_000.0 - player.ecr)
        for player in make_players()
    ]
    python = MonteCarloDraft(context, players, seed=7)
    native = NativeMonteCarloDraft(context, players, seed=7)

    python_candidates = python._ordered_candidates(
        python.base_state,
        context.roster_id,
        python.value_order,
        None,
        None,
        limit=40,
    )[:6]
    native_indices = native._engine.candidate_indices(6)

    assert [player.sleeper_id for player in python_candidates] == [
        native.players[index].sleeper_id for index in native_indices
    ]


@pytest.mark.skipif(not native_available(), reason="native extension is not built")
def test_native_recommendation_allocates_exact_total() -> None:
    native = NativeMonteCarloDraft(make_context(), make_players(), seed=7)

    report = native.recommend(simulations=80, candidate_count=3, workers=2)

    assert report.total_rollouts == 80
    assert sorted(item.samples for item in report.recommendations) == [26, 27, 27]


@pytest.mark.skipif(not native_available(), reason="native extension is not built")
def test_native_results_are_stable_across_worker_counts() -> None:
    context = make_context()
    players = make_players()

    serial = NativeMonteCarloDraft(context, players, seed=29).recommend(300, 3, 1)
    parallel = NativeMonteCarloDraft(context, players, seed=29).recommend(300, 3, 4)

    serial_by_player = {item.player.sleeper_id: item for item in serial.recommendations}
    parallel_by_player = {item.player.sleeper_id: item for item in parallel.recommendations}
    assert serial_by_player.keys() == parallel_by_player.keys()
    for player_id, serial_item in serial_by_player.items():
        parallel_item = parallel_by_player[player_id]
        assert serial_item.samples == parallel_item.samples
        assert serial_item.top_roster_rate == parallel_item.top_roster_rate
        assert serial_item.mean_score == pytest.approx(
            parallel_item.mean_score, rel=1e-12
        )


@pytest.mark.skipif(not native_available(), reason="native extension is not built")
def test_native_distribution_matches_python_reference() -> None:
    context = make_context()
    players = make_players()

    python_report = MonteCarloDraft(context, players, seed=41).recommend(600, 3, 1)
    native_report = NativeMonteCarloDraft(context, players, seed=41).recommend(600, 3, 1)

    python_by_player = {
        item.player.sleeper_id: item for item in python_report.recommendations
    }
    native_by_player = {
        item.player.sleeper_id: item for item in native_report.recommendations
    }
    assert python_by_player.keys() == native_by_player.keys()
    for player_id, python_item in python_by_player.items():
        native_item = native_by_player[player_id]
        assert native_item.mean_score == pytest.approx(
            python_item.mean_score, rel=0.05
        )
        assert native_item.top_roster_rate == pytest.approx(
            python_item.top_roster_rate, abs=0.15
        )


@pytest.mark.skipif(not native_available(), reason="native extension is not built")
def test_native_supports_pre_turn_and_pick_updates() -> None:
    context = make_context()
    native = NativeMonteCarloDraft(context, make_players(), seed=13)
    picks = [
        {
            "pick_no": 1,
            "round": 1,
            "roster_id": 1,
            "player_id": "RB-0",
            "metadata": {"position": "RB"},
        }
    ]

    native.update_picks(picks, seed=14)
    report = native.recommend(120, candidate_count=5, workers=2)

    assert not report.on_clock
    assert report.total_rollouts == 120
    assert sum(item.samples for item in report.recommendations) == 120


def test_auto_engine_warns_and_falls_back_when_native_is_missing(monkeypatch) -> None:
    import sleeper_draft_assistant.native_engine as native_module

    monkeypatch.setattr(native_module, "_NativeDraftEngine", None)
    monkeypatch.setattr(
        native_module, "_NATIVE_IMPORT_ERROR", ImportError("test missing extension")
    )

    with pytest.warns(RuntimeWarning, match="falling back to Python"):
        simulator, engine_name = create_recommendation_engine(
            "auto", make_context(), make_players(), {}, 7
        )

    assert isinstance(simulator, MonteCarloDraft)
    assert engine_name == "python"
    with pytest.raises(NativeEngineUnavailable):
        create_recommendation_engine("cpp", make_context(), make_players(), {}, 7)


def test_flex_assignment_reassigns_players_for_overlapping_slots() -> None:
    candidates = [("wr", "WR", 30.0), ("rb", "RB", 25.0), ("te", "TE", 5.0)]
    assignment = assign_starters(candidates, ("WRRB_FLEX", "REC_FLEX"))

    assert assignment == {0: "rb", 1: "wr"}


def test_draft_superflex_values_second_quarterback_and_fills_lineup() -> None:
    context = make_context(teams=2, rounds=3, positions=("QB", "RB", "SUPER_FLEX"))
    players = make_players()
    simulator = MonteCarloDraft(context, players, seed=7)
    points = {"QB-0": 300.0, "QB-1": 280.0, "RB-0": 200.0}

    assert simulator._lineup_score(list(points), points, include_bench=False) * 17 == 780
    counts = Counter({"QB": 1, "RB": 1})
    assert simulator._can_add(simulator.by_id["QB-1"], 2, counts, Counter())
    result = simulator.recommend(50, 2)
    assert result.recommendations


def test_last_pick_must_fill_flex_and_backup_qb_is_legal_with_bench_space() -> None:
    players = make_players()
    simulator = MonteCarloDraft(make_context(rounds=3, positions=("QB", "RB", "FLEX")), players)
    counts = Counter({"QB": 1, "RB": 1})

    assert not simulator._can_add(simulator.by_id["QB-1"], 2, counts, Counter())
    assert simulator._can_add(simulator.by_id["WR-1"], 2, counts, Counter())
    with_bench = MonteCarloDraft(make_context(rounds=4, positions=("QB", "RB", "FLEX", "BN")), players)
    assert with_bench._can_add(with_bench.by_id["QB-1"], 2, counts, Counter())


def test_second_required_quarterback_has_no_backup_penalty() -> None:
    simulator = MonteCarloDraft(make_context(positions=("QB", "QB", "RB", "WR")), make_players())
    first = simulator._pick_score(simulator.by_id["QB-0"], Counter(), 2, 25.0)
    second = simulator._pick_score(simulator.by_id["QB-1"], Counter({"QB": 1}), 2, 25.0)

    assert first == second


def test_starter_gain_excludes_bench_credit() -> None:
    context = make_context(rounds=2, positions=("QB", "BN"))
    context.picks = [{"pick_no": 1, "roster_id": 1, "player_id": "QB-0"}]
    players = [replace(player, projected_points=300 if player.sleeper_id == "QB-0" else 150) for player in make_players()]
    simulator = MonteCarloDraft(context, players)

    assert simulator.recommendation_metrics(simulator.by_id["QB-1"])[0] == 0.0


def test_simulated_lineup_does_not_see_future_season_outcomes() -> None:
    context = make_context(rounds=2, positions=("QB", "BN"))
    players = [
        Player("starter", "Starter", "QB", "TST", 1, 1, projected_points=300),
        Player("backup", "Backup", "QB", "TST", 2, 1, projected_points=100),
    ]
    simulator = MonteCarloDraft(context, players)
    unexpected_outcomes = {"starter": 100.0, "backup": 400.0}

    assert simulator._lineup_score(["starter", "backup"], unexpected_outcomes, projected_selection=True) * 17 == pytest.approx(132)


def test_zero_projection_is_not_replaced_by_fallback_points() -> None:
    player = Player("out", "Out Player", "RB", "TST", 1, 1, projection_source="Sleeper")
    simulator = MonteCarloDraft(make_context(positions=("RB",)), [player])

    assert simulator._sample_outcome_points()["out"] == 0


def test_back_to_back_pick_has_certain_availability() -> None:
    simulator = MonteCarloDraft(make_context(), make_players())

    assert simulator._next_turn_availability(simulator.players[0], 8) == 1.0


def test_late_recommendations_track_availability_beyond_initial_top_forty() -> None:
    context = make_context(rounds=12, positions=("RB", "WR", "FLEX") + ("BN",) * 9)
    players = make_players(30)
    for pick_number, player in enumerate(players[:41], start=1):
        context.picks.append({"pick_no": pick_number, "roster_id": roster_for_pick(context, pick_number), "player_id": player.sleeper_id})
    simulator = MonteCarloDraft(context, players, seed=9)
    report = simulator.recommend(20)

    assert report.recommendations
    assert all(item.availability_rate >= item.selection_rate > 0 for item in report.recommendations)


@pytest.mark.parametrize("engine", ["python", "cpp"])
def test_completed_draft_returns_no_recommendations(engine) -> None:
    if engine == "cpp" and not native_available():
        pytest.skip("native extension is not built")
    context = make_context(teams=2, rounds=1, positions=("RB",))
    context.picks = [
        {"pick_no": 1, "roster_id": 1, "player_id": "RB-0"},
        {"pick_no": 2, "roster_id": 2, "player_id": "RB-1"},
    ]
    simulator, _ = create_recommendation_engine(engine, context, make_players(), {}, 7)

    report = simulator.recommend(20)
    assert report.recommendations == ()
    assert report.total_rollouts == 0
    assert not report.on_clock


def test_future_keeper_does_not_skip_earlier_open_picks() -> None:
    context = make_context()
    context.picks = [{"pick_no": 8, "round": 2, "roster_id": 1, "player_id": "RB-0"}]
    simulator = MonteCarloDraft(context, make_players())

    assert simulator.base_state.next_pick == 1
    order, ranks = simulator._sample_opponent_board()
    result = simulator._finish_rollout(simulator.base_state.clone(), order, ranks)
    assert len(result.user_picks) == 4
    assert sum(player_id == "RB-0" for _, player_id in result.user_picks) == 1


def test_linear_and_third_round_reversal_follow_configured_order() -> None:
    context = make_context()
    context.draft["type"] = "linear"
    assert [roster_for_pick(context, pick) for pick in range(5, 9)] == [1, 2, 3, 4]
    context.draft["type"] = "snake"
    context.draft["settings"]["reversal_round"] = 3
    assert [roster_for_pick(context, pick) for pick in range(9, 17)] == [4, 3, 2, 1, 1, 2, 3, 4]


def test_python_candidates_use_common_samples_across_worker_counts() -> None:
    serial = MonteCarloDraft(make_context(), make_players(), seed=91).recommend(80, 3, 1)
    parallel = MonteCarloDraft(make_context(), make_players(), seed=91).recommend(80, 3, 2)
    assert serial.total_rollouts == parallel.total_rollouts == 80
    serial_by_id = {row.player.sleeper_id: row for row in serial.recommendations}
    for row in parallel.recommendations:
        assert row.mean_score == pytest.approx(serial_by_id[row.player.sleeper_id].mean_score)
        assert row.samples == serial_by_id[row.player.sleeper_id].samples


@pytest.mark.skipif(not native_available(), reason="native extension is not built")
def test_native_complex_flex_automatically_uses_python() -> None:
    context = make_context(positions=("QB", "RB", "WR", "SUPER_FLEX"))
    with pytest.warns(RuntimeWarning, match="requires the Python engine"):
        simulator, engine = create_recommendation_engine("auto", context, make_players(), {}, 9)
    assert engine == "python"
    assert isinstance(simulator, MonteCarloDraft)


@pytest.mark.skipif(not native_available(), reason="native extension is not built")
def test_native_late_board_availability_tracks_all_players() -> None:
    context = make_context(rounds=12, positions=("RB", "WR", "FLEX") + ("BN",) * 9)
    players = make_players(30)
    for pick_number, player in enumerate(players[:41], start=1):
        context.picks.append({"pick_no": pick_number, "roster_id": roster_for_pick(context, pick_number), "player_id": player.sleeper_id})
    report = NativeMonteCarloDraft(context, players, seed=9).recommend(20)
    assert all(item.availability_rate >= item.selection_rate > 0 for item in report.recommendations)
