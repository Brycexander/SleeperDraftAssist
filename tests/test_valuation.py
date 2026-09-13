from __future__ import annotations

import polars as pl
import pytest

from sleeper_draft_assistant.models import LeagueRules, Player
from sleeper_draft_assistant.valuation import (
    HistoricalPerformance,
    RawProjection,
    build_player_values,
    load_historical_performance,
    score_projection,
)


def test_score_projection_uses_league_ppr_and_six_point_passing_tds() -> None:
    quarterback = RawProjection(
        name="Test Quarterback",
        team="TST",
        position="QB",
        bye=9,
        stats={
            "pass_yd": 4000,
            "pass_td": 30,
            "pass_int": 12,
            "rush_yd": 500,
            "rush_td": 5,
        },
    )
    running_back = RawProjection(
        name="Test Running Back",
        team="TST",
        position="RB",
        bye=9,
        stats={
            "rush_yd": 1000,
            "rush_td": 10,
            "rec": 60,
            "rec_yd": 500,
            "rec_td": 4,
        },
    )
    scoring = {
        "pass_yd": 0.04,
        "pass_td": 6,
        "pass_int": -2,
        "rush_yd": 0.1,
        "rush_td": 6,
        "rec": 1,
        "rec_yd": 0.1,
        "rec_td": 6,
    }

    assert score_projection(quarterback, scoring) == 396
    assert score_projection(running_back, scoring) == 294


def test_build_player_values_uses_replacement_level_and_market_rank(
    monkeypatch, tmp_path
) -> None:
    players = [
        Player("rb-1", "Alpha Back", "RB", "AAA", 2.0, 1.0),
        Player("rb-2", "Beta Back", "RB", "BBB", 12.0, 2.0),
        Player("rb-3", "Gamma Back", "RB", "CCC", 30.0, 3.0),
    ]
    projections = [
        RawProjection("Alpha Back", "AAA", "RB", 8, {"rush_yd": 2000}),
        RawProjection("Beta Back", "BBB", "RB", 9, {"rush_yd": 1000}),
        RawProjection("Gamma Back", "CCC", "RB", 10, {"rush_yd": 500}),
    ]
    metadata = {
        player.sleeper_id: {
            "team": player.team,
            "depth_chart_order": 1,
            "search_rank": index,
            "years_exp": 2,
            "age": 25,
        }
        for index, player in enumerate(players, start=1)
    }
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_sleeper_projections",
        lambda *args, **kwargs: (projections, {"rb-1": 7.5}),
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_sleeper_metadata",
        lambda *args, **kwargs: metadata,
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_public_sleeper_adp",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_historical_performance",
        lambda *args, **kwargs: {},
    )
    rules = LeagueRules(
        teams=1,
        rounds=3,
        roster_positions=("RB", "BN", "BN"),
        scoring={"rush_yd": 0.1},
    )

    result = build_player_values(players, rules, tmp_path, 2026)
    alpha, beta, gamma = result.players

    assert [alpha.name, beta.name, gamma.name] == [
        "Alpha Back",
        "Beta Back",
        "Gamma Back",
    ]
    assert alpha.projected_points == 200
    assert alpha.vorp == 100
    assert beta.vorp == 0
    assert gamma.vorp == -50
    assert alpha.adp == 7.5
    assert alpha.online_projected_points == 200
    assert alpha.historical_points == 0
    assert alpha.value_score == pytest.approx(
        alpha.projection_component + alpha.consensus_component - alpha.risk_penalty
    )
    assert result.diagnostics.replacement_points["RB"] == 100
    assert result.diagnostics.adp_counts == {
        "Sleeper projection ADP": 1,
        "ECR fallback": 2,
    }


def test_flex_replacement_goes_to_highest_projected_remaining_player(
    monkeypatch, tmp_path
) -> None:
    players = [
        Player("rb-1", "RB One", "RB", "AAA", 1.0, 1.0),
        Player("rb-2", "RB Two", "RB", "AAA", 4.0, 1.0),
        Player("rb-3", "RB Three", "RB", "AAA", 6.0, 1.0),
        Player("wr-1", "WR One", "WR", "BBB", 2.0, 1.0),
        Player("wr-2", "WR Two", "WR", "BBB", 3.0, 1.0),
        Player("wr-3", "WR Three", "WR", "BBB", 5.0, 1.0),
    ]
    points = {
        "rb-1": 300,
        "rb-2": 100,
        "rb-3": 50,
        "wr-1": 250,
        "wr-2": 240,
        "wr-3": 10,
    }
    projections = [
        RawProjection(
            player.name,
            player.team,
            player.position,
            None,
            {"rush_yd" if player.position == "RB" else "rec_yd": points[player.sleeper_id]},
            sleeper_id=player.sleeper_id,
        )
        for player in players
    ]
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_sleeper_projections",
        lambda *args, **kwargs: (projections, {}),
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_sleeper_metadata",
        lambda *args, **kwargs: {
            player.sleeper_id: {"team": player.team, "depth_chart_order": 1}
            for player in players
        },
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_public_sleeper_adp",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_historical_performance",
        lambda *args, **kwargs: {},
    )
    rules = LeagueRules(
        teams=1,
        rounds=3,
        roster_positions=("RB", "WR", "FLEX"),
        scoring={"rush_yd": 1, "rec_yd": 1},
    )

    result = build_player_values(players, rules, tmp_path, 2026)

    assert result.diagnostics.replacement_points["RB"] == 100
    assert result.diagnostics.replacement_points["WR"] == 10


def test_historical_performance_uses_league_scoring_and_recency(
    monkeypatch, tmp_path
) -> None:
    stats = pl.DataFrame(
        {
            "player_id": ["gsis-1", "gsis-1"],
            "season": [2024, 2025],
            "position": ["RB", "RB"],
            "games": [10, 10],
            "carries": [100, 100],
            "rushing_yards": [500, 1_000],
            "rushing_tds": [5, 10],
            "receptions": [20, 40],
            "receiving_yards": [200, 500],
            "receiving_tds": [2, 4],
        }
    )
    player_ids = pl.DataFrame(
        {"gsis_id": ["gsis-1"], "sleeper_id": [1234]}
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation._load_historical_frames",
        lambda *args, **kwargs: (stats, player_ids),
    )
    rules = LeagueRules(
        teams=10,
        rounds=15,
        roster_positions=("RB", "BN"),
        scoring={
            "rush_yd": 0.1,
            "rush_td": 6,
            "rec": 1,
            "rec_yd": 0.1,
            "rec_td": 6,
        },
    )

    result = load_historical_performance(tmp_path, 2026, rules)

    assert set(result) == {"1234"}
    assert result["1234"].games == 20
    assert result["1234"].seasons == 2
    assert result["1234"].season_equivalent_points > 200
    assert 0 < result["1234"].reliability < 1


def test_player_projection_blends_reliable_historical_performance(
    monkeypatch, tmp_path
) -> None:
    player = Player("rb-1", "Alpha Back", "RB", "AAA", 1.0, 1.0)
    projection = RawProjection(
        "Alpha Back", "AAA", "RB", 8, {"rush_yd": 2_000}
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_sleeper_projections",
        lambda *args, **kwargs: ([projection], {}),
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_sleeper_metadata",
        lambda *args, **kwargs: {
            "rb-1": {"team": "AAA", "depth_chart_order": 1}
        },
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_public_sleeper_adp",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "sleeper_draft_assistant.valuation.load_historical_performance",
        lambda *args, **kwargs: {
            "rb-1": HistoricalPerformance(100.0, 34, 2, 1.0)
        },
    )
    rules = LeagueRules(
        teams=1,
        rounds=1,
        roster_positions=("RB",),
        scoring={"rush_yd": 0.1},
    )

    valued = build_player_values([player], rules, tmp_path, 2026).players[0]

    assert valued.online_projected_points == 200
    assert valued.historical_points == 100
    assert 0.20 < valued.history_weight <= 0.25
    assert 175 < valued.projected_points < 180


@pytest.mark.parametrize("include_bonus_stat", [False, True])
def test_sleeper_reception_premium_is_applied_exactly_once(include_bonus_stat) -> None:
    stats = {"rec": 60.0, "rec_yd": 800.0}
    if include_bonus_stat:
        stats["bonus_rec_te"] = 60.0
    projection = RawProjection("Test Tight End", "TST", "TE", None, stats, source="Sleeper")

    assert score_projection(projection, {"rec": 1.0, "rec_yd": 0.1, "bonus_rec_te": 0.5}) == 170.0


def test_zero_projected_player_is_not_imputed_or_revived_by_history(monkeypatch, tmp_path) -> None:
    player = Player("rb-1", "Injured Back", "RB", "AAA", 1.0, 1.0)
    projection = RawProjection(player.name, "AAA", "RB", None, {"rush_yd": 0.0}, source="Sleeper", sleeper_id="rb-1")
    monkeypatch.setattr("sleeper_draft_assistant.valuation.load_sleeper_projections", lambda *args, **kwargs: ([projection], {}))
    monkeypatch.setattr("sleeper_draft_assistant.valuation.load_sleeper_metadata", lambda *args, **kwargs: {"rb-1": {"team": "AAA"}})
    monkeypatch.setattr("sleeper_draft_assistant.valuation.load_public_sleeper_adp", lambda *args, **kwargs: {})
    monkeypatch.setattr("sleeper_draft_assistant.valuation.load_historical_performance", lambda *args, **kwargs: {"rb-1": HistoricalPerformance(250.0, 34, 2, 1.0)})
    rules = LeagueRules(teams=1, rounds=1, roster_positions=("RB",), scoring={"rush_yd": 0.1})

    valued = build_player_values([player], rules, tmp_path, 2026).players[0]
    assert valued.projected_points == 0.0
    assert valued.projection_source == "Sleeper"


def test_superflex_replacement_accounts_for_second_quarterback(monkeypatch, tmp_path) -> None:
    players = [
        Player("qb-1", "QB One", "QB", "AAA", 1, 1),
        Player("qb-2", "QB Two", "QB", "AAA", 2, 1),
        Player("qb-3", "QB Three", "QB", "AAA", 3, 1),
        Player("rb-1", "RB One", "RB", "AAA", 4, 1),
        Player("rb-2", "RB Two", "RB", "AAA", 5, 1),
    ]
    points = [300.0, 280.0, 200.0, 180.0, 100.0]
    projections = [RawProjection(player.name, "AAA", player.position, None, {"rush_yd": point}, sleeper_id=player.sleeper_id) for player, point in zip(players, points)]
    monkeypatch.setattr("sleeper_draft_assistant.valuation.load_sleeper_projections", lambda *args, **kwargs: (projections, {}))
    monkeypatch.setattr("sleeper_draft_assistant.valuation.load_sleeper_metadata", lambda *args, **kwargs: {player.sleeper_id: {"team": "AAA", "depth_chart_order": 1} for player in players})
    monkeypatch.setattr("sleeper_draft_assistant.valuation.load_public_sleeper_adp", lambda *args, **kwargs: {})
    monkeypatch.setattr("sleeper_draft_assistant.valuation.load_historical_performance", lambda *args, **kwargs: {})
    rules = LeagueRules(teams=1, rounds=3, roster_positions=("QB", "RB", "SUPER_FLEX"), scoring={"rush_yd": 1.0})

    result = build_player_values(players, rules, tmp_path, 2026)
    assert result.diagnostics.replacement_points["QB"] == 200.0
    assert result.diagnostics.replacement_points["RB"] == 100.0
