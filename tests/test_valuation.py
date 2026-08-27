from __future__ import annotations

from sleeper_draft_assistant.models import LeagueRules, Player
from sleeper_draft_assistant.valuation import (
    RawProjection,
    build_player_values,
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
    rules = LeagueRules(
        teams=1,
        rounds=3,
        roster_positions=("RB", "WR", "FLEX"),
        scoring={"rush_yd": 1, "rec_yd": 1},
    )

    result = build_player_values(players, rules, tmp_path, 2026)

    assert result.diagnostics.replacement_points["RB"] == 100
    assert result.diagnostics.replacement_points["WR"] == 10
