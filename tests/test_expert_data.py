from datetime import datetime, timezone
import math
import pytest

from sleeper_draft_assistant.expert_data import (
    AccuracyEvidence,
    ExpertRow,
    ExpertSubmission,
    build_consensus,
    rank_credit,
    select_expert_panel,
)


def sub(expert, a, b):
    return ExpertSubmission(expert_id=expert, name=expert, publisher="fixture",
        published_at=datetime.now(timezone.utc), source_url="https://example.invalid/ranks",
        accuracy_history=[
            AccuracyEvidence(season=2025, scoring="PPR", place=1, field_size=100, graded_weeks=14, url="https://example.invalid/accuracy"),
            AccuracyEvidence(season=2024, scoring="PPR", place=2, field_size=100, graded_weeks=14, url="https://example.invalid/accuracy"),
        ],
        rows=[ExpertRow(player_id="a", position="WR", overall_rank=a), ExpertRow(player_id="b", position="WR", overall_rank=b)])


def test_median_consensus_and_credit():
    board = build_consensus([sub("e1", 1, 10), sub("e2", 2, 11), sub("e3", 90, 12)])
    assert board.values["a"].median_rank == 2
    assert board.values["a"].worst_rank == 90
    assert board.values["a"].rank_credit > board.values["b"].rank_credit
    assert rank_credit(10, 100) == pytest.approx(math.log(101 / 10))


def test_duplicate_and_small_panel_rejected():
    with pytest.raises(ValueError):
        build_consensus([sub("e1", 1, 2), sub("e1", 1, 2), sub("e2", 1, 2)])
    with pytest.raises(ValueError):
        build_consensus([sub("e1", 1, 2), sub("e2", 1, 2)])


def candidate(expert, publisher, places, *, published_at=None, graded_weeks=14):
    history = [
        AccuracyEvidence(
            season=season,
            scoring="PPR",
            place=place,
            field_size=100,
            graded_weeks=graded_weeks,
            url="https://example.invalid/accuracy",
        )
        for season, place in places
    ]
    return ExpertSubmission(
        expert_id=expert,
        name=expert,
        publisher=publisher,
        published_at=published_at or datetime.now(timezone.utc),
        source_url="https://example.invalid/ranks",
        accuracy_history=history,
        rows=[ExpertRow(player_id=f"p{i}", position=("QB", "RB", "WR", "TE")[i % 4], overall_rank=i + 1) for i in range(100)],
    )


def test_selects_five_to_eight_by_rolling_accuracy_with_publisher_cap():
    submissions = [
        candidate("a1", "A", [(2025, 1), (2024, 1), (2023, 1)]),
        candidate("a2", "A", [(2025, 2), (2024, 2), (2023, 2)]),
        candidate("a3", "A", [(2025, 3), (2024, 3), (2023, 3)]),
        candidate("b1", "B", [(2025, 4), (2024, 4), (2023, 4)]),
        candidate("c1", "C", [(2025, 5), (2024, 5), (2023, 5)]),
        candidate("d1", "D", [(2025, 6), (2024, 6), (2023, 6)]),
        candidate("e1", "E", [(2025, 7), (2024, 7), (2023, 7)]),
        candidate("f1", "F", [(2025, 8), (2024, 8), (2023, 8)]),
        candidate("g1", "G", [(2025, 9), (2024, 9), (2023, 9)]),
    ]

    selection = select_expert_panel(
        submissions, season=2026, week=5, scoring="PPR", now=datetime.now(timezone.utc)
    )

    assert [item.expert_id for item in selection.selected] == ["a1", "a2", "b1", "c1", "d1", "e1", "f1", "g1"]
    assert any(item.expert_id == "a3" and item.reason == "publisher_limit" for item in selection.excluded)
    assert selection.scores["a1"] > selection.scores["b1"]


def test_requires_two_completed_ros_seasons_with_enough_graded_weeks():
    submissions = [
        candidate("one-season", "A", [(2025, 1)]),
        candidate("too-few-weeks", "B", [(2025, 1), (2024, 1)], graded_weeks=5),
    ]

    selection = select_expert_panel(
        submissions, season=2026, week=5, scoring="PPR", now=datetime.now(timezone.utc)
    )

    assert not selection.selected
    assert {item.reason for item in selection.excluded} == {"insufficient_history", "insufficient_graded_weeks"}


def test_optional_thin_third_season_does_not_disqualify_two_complete_seasons():
    expert = candidate("qualified", "A", [(2025, 3), (2024, 4), (2023, 1)])
    expert.accuracy_history[-1].graded_weeks = 5

    selection = select_expert_panel(
        [expert], season=2026, week=5, scoring="PPR", now=datetime.now(timezone.utc), minimum=1
    )

    assert [item.expert_id for item in selection.selected] == ["qualified"]


def test_two_season_record_is_shrunk_and_current_season_is_capped_at_twenty_percent():
    two_year = candidate("two", "A", [(2025, 1), (2024, 1)])
    three_year = candidate("three", "B", [(2025, 10), (2024, 10), (2023, 10)])
    current_star = candidate("current", "C", [(2026, 1), (2025, 20), (2024, 20), (2023, 20)])
    current_star.accuracy_history[0].graded_weeks = 12

    early = select_expert_panel(
        [two_year, three_year, current_star], season=2026, week=6, scoring="PPR", now=datetime.now(timezone.utc), minimum=1
    )
    later = select_expert_panel(
        [two_year, three_year, current_star], season=2026, week=13, scoring="PPR", now=datetime.now(timezone.utc), minimum=1
    )

    assert early.scores["two"] < early.scores["three"]
    assert later.scores["current"] > early.scores["current"]
    historical = early.scores["current"]
    assert later.scores["current"] == pytest.approx(historical * .8 + 1.0 * .2)


def test_completed_seasons_use_fifty_thirty_twenty_recency_weights():
    mixed = candidate("mixed", "A", [(2025, 1), (2024, 100), (2023, 100)])

    selection = select_expert_panel(
        [mixed], season=2026, week=5, scoring="PPR", now=datetime.now(timezone.utc), minimum=1
    )

    assert selection.scores["mixed"] == pytest.approx(.5)


def test_current_season_weight_cannot_run_ahead_of_completed_weeks():
    expert = candidate("current", "A", [(2026, 1), (2025, 20), (2024, 20), (2023, 20)])
    expert.accuracy_history[0].graded_weeks = 18
    historical = select_expert_panel(
        [expert], season=2026, week=6, scoring="PPR", now=datetime.now(timezone.utc), minimum=1
    ).scores["current"]

    week_seven = select_expert_panel(
        [expert], season=2026, week=7, scoring="PPR", now=datetime.now(timezone.utc), minimum=1
    ).scores["current"]

    assert week_seven == pytest.approx(historical * .9 + 1.0 * .1)


def test_consensus_reports_player_rank_disagreement():
    board = build_consensus([
        sub("e1", 1, 10), sub("e2", 2, 11), sub("e3", 90, 12), sub("e4", 3, 13), sub("e5", 4, 14)
    ])

    assert board.values["a"].rank_spread == 89
    assert board.values["a"].agreement == "low"
    assert board.values["b"].agreement == "high"
