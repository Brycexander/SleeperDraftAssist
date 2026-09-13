from dataclasses import replace
from itertools import permutations
import json
import math
import random

import pytest

from sleeper_draft_assistant.strategy import WeeklyPlayer, optimize_lineup, suggest_trades, suggest_waivers


def player(player_id, position, points, **kwargs):
    kwargs.setdefault("roster_value", max(0, points) if points is not None else 0)
    return WeeklyPlayer(player_id, player_id, position, "TST", points, **kwargs)


def starter_ids(result):
    return [p["player_id"] for p in result["starters"]]


def test_exact_assignment_preserves_flex_and_superflex_eligibility():
    players = [player("rb1", "RB", 20), player("rb2", "RB", 19),
               player("wr", "WR", 22), player("qb1", "QB", 30), player("qb2", "QB", 28)]
    result = optimize_lineup(players, ("FLEX", "RB", "QB", "SUPER_FLEX"))
    assert result["projected_points"] == 100
    assert starter_ids(result) == ["wr", "rb1", "qb1", "qb2"]
    assert len(set(starter_ids(result))) == 4


def test_assignment_supports_multi_position_players_and_defense_aliases():
    result = optimize_lineup([
        player("dual", "RB", 20, eligible_positions=("RB", "WR")),
        player("rb", "RB", 19), player("wr", "WR", 1), player("dst", "D/ST", 5),
    ], ("RB", "WR", "DEF"))
    assert starter_ids(result) == ["rb", "dual", "dst"]
    assert result["projected_points"] == 44


def test_assignment_matches_brute_force_for_random_small_rosters():
    randomizer = random.Random(8823)
    slots = ("RB", "WR", "FLEX", "SUPER_FLEX")
    allowed = ({"RB"}, {"WR"}, {"RB", "WR", "TE"}, {"QB", "RB", "WR", "TE"})
    for _ in range(15):
        players = [player(str(i), position, randomizer.uniform(1, 30)) for i, position in enumerate(("RB", "RB", "WR", "WR", "TE", "QB"))]
        expected = max(sum(p.points for p in lineup) for lineup in permutations(players, len(slots))
            if all(p.position in permitted for p, permitted in zip(lineup, allowed)))
        assert optimize_lineup(players, slots)["projected_points"] == pytest.approx(expected, abs=0.0001)


def test_locks_preserve_original_slot_and_locked_bench_even_with_empty_slot_marker():
    players = [player("low", "WR", 3, locked=True), player("locked_bench", "RB", 40, locked=True),
               player("rb", "RB", 12), player("wr", "WR", 25)]
    result = optimize_lineup(players, ("RB", "FLEX", "WR"), current_starters=["0", "low", "wr"])
    assert starter_ids(result) == ["rb", "low", "wr"]
    assert result["bench"][0]["player_id"] == "locked_bench"


def test_locked_injured_starter_stays_but_unavailable_unlocked_players_are_excluded():
    result = optimize_lineup([
        player("locked", "RB", None, status="Out", locked=True),
        player("ir", "RB", 50, status="IR"), player("bye", "RB", 45, bye=True),
        player("available", "RB", 10),
    ], ("RB", "FLEX"), current_starters=["locked", "0"])
    assert starter_ids(result) == ["locked", "available"]
    assert result["projected_points"] == 10
    assert result["missing_projection_count"] == 1
    assert any("known subtotal" in warning for warning in result["warnings"])


def test_missing_forecasts_are_not_invented_and_duplicate_players_cannot_fill_twice():
    known = player("known", "RB", 10)
    result = optimize_lineup([known, known, player("unknown", "WR", None), player("nan", "WR", math.nan)], ("RB", "FLEX", "WR"))
    assert starter_ids(result) == ["known", None, None]
    assert result["projected_points"] == 10
    assert result["missing_projection_count"] == 2
    assert result["unfilled_slots"] == 2
    json.dumps(result, allow_nan=False)


def test_zero_points_can_start_but_negative_projection_can_remain_empty():
    assert starter_ids(optimize_lineup([player("zero", "DEF", 0)], ("DEF",))) == ["zero"]
    assert starter_ids(optimize_lineup([player("negative", "DEF", -2)], ("DEF",))) == [None]


def test_restricted_flex_slots_do_not_accept_qb_or_wrong_position():
    result = optimize_lineup([player("qb", "QB", 50), player("rb", "RB", 20),
        player("wr", "WR", 10), player("te", "TE", 15)], ("REC_FLEX", "WRRB_FLEX"))
    assert starter_ids(result) == ["te", "rb"]


def test_tiers_reorder_same_position_without_comparing_raw_tiers_in_flex():
    # A WR's tier 5 has no numeric relation to a RB's tier 1.
    cross_position = optimize_lineup([player("rb", "RB", 10, tier=1), player("wr", "WR", 20, tier=5)], ("FLEX",), "tiers")
    assert starter_ids(cross_position) == ["wr"]
    same_position = optimize_lineup([player("a", "RB", 20, tier=3), player("b", "RB", 15, tier=1)], ("RB",), "tiers")
    assert starter_ids(same_position) == ["b"]
    assert same_position["projected_points"] == 15
    assert same_position["objective_score"] == 20


def test_shared_flex_ranks_can_compare_positions():
    result = optimize_lineup([player("rb", "RB", 10, flex_rank=1), player("wr", "WR", 20, flex_rank=2)], ("FLEX",), "tiers")
    assert starter_ids(result) == ["rb"]
    assert result["projected_points"] == 10


def test_waivers_exclude_reserves_taxi_and_all_other_rosters():
    values = [player("mine", "RB", 5), player("free", "RB", 10), player("other", "RB", 20),
              player("reserve", "RB", 30), player("taxi", "RB", 35), player("starter_only", "RB", 40)]
    rosters = [{"roster_id": 1, "players": ["mine"], "starters": ["mine"]},
               {"roster_id": 2, "players": ["other"], "reserve": ["reserve"], "taxi": ["taxi"], "starters": ["starter_only"]}]
    result = suggest_waivers({p.player_id: p for p in values}, rosters, 1, ("RB",), capacity=1)
    assert [r["add"]["player_id"] for r in result] == ["free"]
    assert result[0]["drop"]["player_id"] == "mine"
    assert result[0]["weekly_gain"] == 5


def test_waivers_use_empty_roster_capacity_without_an_unnecessary_drop():
    values = [player("mine", "RB", 5), player("free", "RB", 10)]
    result = suggest_waivers({p.player_id: p for p in values}, [{"roster_id": 1, "players": ["mine"]}], 1, ("RB",), capacity=2)
    assert result[0]["drop"] is None
    assert result[0]["depth_gain"] == pytest.approx(0.6)


def test_waivers_protect_season_value_of_bye_and_injured_stars():
    values = [player("star", "RB", 0, bye=True, roster_value=25),
              player("hurt", "RB", None, status="Out", roster_value=24),
              player("low", "RB", 8, roster_value=10), player("free", "RB", 12, roster_value=12)]
    result = suggest_waivers({p.player_id: p for p in values}, [{"roster_id": 1, "players": ["star", "hurt", "low"]}], 1, ("RB",), capacity=3)
    assert len(result) == 1
    assert result[0]["drop"]["player_id"] == "low"


def test_waivers_do_not_drop_injured_or_missing_forecast_player_without_long_term_data():
    values = [player("hurt", "RB", None, status="Out"), player("free", "RB", 20)]
    assert suggest_waivers({p.player_id: p for p in values}, [{"roster_id": 1, "players": ["hurt"]}], 1, ("RB",), capacity=1) == []


def test_missing_season_value_protects_healthy_players_from_drops_and_trades():
    values = [player("star", "RB", 4, roster_value=0), player("free", "RB", 20)]
    data = {p.player_id: p for p in values}
    roster = [{"roster_id": 1, "players": ["star"]}]
    assert suggest_waivers(data, roster, 1, ("RB",), capacity=1) == []
    assert suggest_waivers(data, roster, 1, ("RB",), capacity=2)[0]["drop"] is None
    players, rosters = balanced_league()
    players = {key: replace(p, roster_value=0) for key, p in players.items()}
    assert suggest_trades(players, rosters, 1, ("RB", "WR")) == []


def test_waivers_cannot_drop_or_add_locked_players():
    values = [player("mine", "RB", 5, locked=True), player("free", "RB", 10), player("played", "RB", 40, locked=True)]
    data = {p.player_id: p for p in values}
    assert suggest_waivers(data, [{"roster_id": 1, "players": ["mine"], "starters": ["mine"]}], 1, ("RB",), capacity=1) == []


def balanced_league():
    values = [player("own_rb1", "RB", 18, roster_value=18), player("own_rb2", "RB", 16, roster_value=16),
              player("own_wr", "WR", 6, roster_value=6), player("their_wr1", "WR", 18, roster_value=18),
              player("their_wr2", "WR", 16, roster_value=16), player("their_rb", "RB", 6, roster_value=6)]
    rosters = [{"roster_id": 1, "players": ["own_rb1", "own_rb2", "own_wr"], "starters": ["own_rb1", "own_wr"]},
               {"roster_id": 2, "players": ["their_wr1", "their_wr2", "their_rb"], "starters": ["their_rb", "their_wr1"]}]
    return {p.player_id: p for p in values}, rosters


def test_trades_find_balanced_mutual_starter_improvements():
    players, rosters = balanced_league()
    suggestions = suggest_trades(players, rosters, 1, ("RB", "WR"))
    swap = next(s for s in suggestions if [p["player_id"] for p in s["give"]] == ["own_rb2"] and [p["player_id"] for p in s["receive"]] == ["their_wr2"])
    assert swap["weekly_gain"] == 10
    assert swap["partner_weekly_gain"] == 10
    assert swap["total_gain"] > 0
    assert swap["partner_total_gain"] > 0
    assert swap["value_ratio"] == 1
    json.dumps(suggestions, allow_nan=False)


def test_trades_reject_offers_that_leave_other_team_without_a_required_starter():
    players, rosters = balanced_league()
    rosters[1]["players"] = ["their_wr2", "their_rb"]
    rosters[1]["starters"] = ["their_rb", "their_wr2"]
    suggestions = suggest_trades(players, rosters, 1, ("RB", "WR"))
    assert not any(len(s["give"]) == 1 and s["give"][0]["position"] == "RB"
                   and s["receive"][0]["player_id"] == "their_wr2" for s in suggestions)
    # Two-for-two deals may still help both rosters while filling both slots.
    for suggestion in suggestions:
        after_ids = (set(rosters[1]["players"]) - {p["player_id"] for p in suggestion["receive"]}) | {p["player_id"] for p in suggestion["give"]}
        assert optimize_lineup([players[p] for p in after_ids], ("RB", "WR"))["unfilled_slots"] == 0


def test_trades_exclude_frozen_reserve_and_taxi_assets():
    players, rosters = balanced_league()
    players["own_rb1"] = replace(players["own_rb1"], locked=True)
    rosters[0]["taxi"] = ["own_rb2"]
    assert suggest_trades(players, rosters, 1, ("RB", "WR")) == []


def test_bad_mode_and_unknown_roster_fail_clearly():
    with pytest.raises(ValueError, match="mode"):
        optimize_lineup([], (), "made-up")
    with pytest.raises(ValueError, match="not found"):
        suggest_waivers({}, [], 999, ())
