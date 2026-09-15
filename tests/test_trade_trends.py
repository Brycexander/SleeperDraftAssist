from dataclasses import replace
import json

import pytest

from sleeper_draft_assistant import strategy


def player(pid, value=15, points=15, **kwargs):
    return strategy.WeeklyPlayer(pid, pid, "WR", "CHI", points, roster_value=value, **kwargs)


def games(points, work=8):
    return [{"week": i + 1, "points": p, "opportunities": work,
             "snap_share": .8, "team": "CHI"} for i, p in enumerate(points)]


def trend(p, rows):
    assert hasattr(strategy, "analyze_trade_trend"), "Recent-form analysis is not implemented"
    return strategy.analyze_trade_trend(p, rows)


def test_single_appearance_is_provisional_and_later_isolated_spike_is_not_a_streak():
    one = trend(player("one"), games([35]))
    assert one["signal"] == "sell_high"
    assert one["provisional"] is True
    assert one["confidence"] == "very_low"
    assert one["workload_change"] is None
    assert trend(player("two"), games([30, 30]))["signal"] == "sell_high"
    assert trend(player("spike"), games([15, 15, 45]))["signal"] == "neutral"


def test_stable_workload_separates_hot_perception_from_cold_upside():
    hot = trend(player("hot", value=12, points=12), games([23, 24, 25, 24]))
    cold = trend(player("cold", value=18, points=18), games([9, 10, 8, 9]))
    assert hot["signal"] == "sell_high"
    assert cold["signal"] == "buy_low"
    assert hot["perceived_value"] > hot["outlook_value"]
    assert cold["perceived_value"] < cold["outlook_value"]
    assert cold["outlook_value"] > hot["outlook_value"]
    assert cold["games"] == 4
    json.dumps(cold, allow_nan=False)


def test_workload_growth_is_breakout_and_decline_is_not_automatic_buy_low():
    rising = games([25, 25, 25, 25], work=5)
    falling = games([8, 8, 8, 8], work=10)
    for row in rising[2:]: row["opportunities"] = 10
    for row in falling[2:]: row["opportunities"] = 4
    assert trend(player("up"), rising)["signal"] == "possible_breakout"
    assert trend(player("down"), falling)["signal"] == "role_decline"


def test_injuries_missing_usage_and_team_changes_do_not_create_false_bargains():
    rows = games([7, 8, 7, 8])
    assert trend(player("hurt", status="Out"), rows)["signal"] == "unavailable"
    assert trend(replace(player("newteam"), team="DAL"), rows)["signal"] == "role_change"
    for row in rows: row["opportunities"] = None
    assert trend(player("unknown"), rows)["signal"] == "insufficient_data"


def test_buy_low_can_improve_outlook_despite_losing_points_this_week():
    hot, cold = player("hot", 12, 19), player("cold", 18, 10)
    players = {p.player_id: p for p in (hot, cold)}
    trends = {"hot": trend(hot, games([20, 19, 20, 19])),
              "cold": trend(cold, games([11, 12, 11, 12]))}
    assert hasattr(strategy, "suggest_opportunity_trades"), "Opportunity trade search is not implemented"
    results = strategy.suggest_opportunity_trades(players, [
        {"roster_id": 1, "players": ["hot"], "starters": ["hot"]},
        {"roster_id": 2, "players": ["cold"], "starters": ["cold"]},
    ], 1, ("WR",), trends)
    assert len(results) == 1
    assert results[0]["weekly_gain"] == -9
    assert results[0]["outlook_gain"] > 0
    assert results[0]["partner_perceived_gain"] > 0
    assert results[0]["partner_outlook_gain"] < 0
    assert results[0]["kind"] == "buy_low_sell_high"
    assert results[0]["warnings"]


def test_opportunity_search_rejects_missing_streaks_and_locked_assets():
    hot, cold = player("hot", 12, 19, locked=True), player("cold", 18, 10)
    players = {p.player_id: p for p in (hot, cold)}
    trends = {"hot": trend(hot, games([27, 26, 25, 26])), "cold": trend(cold, games([7, 8, 7, 8]))}
    assert hasattr(strategy, "suggest_opportunity_trades")
    rosters = [{"roster_id": 1, "players": ["hot"]}, {"roster_id": 2, "players": ["cold"]}]
    assert strategy.suggest_opportunity_trades(players, rosters, 1, ("WR",), trends) == []


def test_trend_rejects_nonfinite_or_duplicate_games():
    rows = games([8, 8, 8])
    rows[2]["week"] = 2
    assert trend(player("duplicate"), rows)["games"] == 2
    rows = games([8, 8, float("nan")])
    assert trend(player("nan"), rows)["games"] == 2


def test_one_game_can_generate_a_cautious_buy_low_sell_high_trade():
    hot, cold = player("hot", 14, 14), player("cold", 16, 16)
    players = {p.player_id: p for p in (hot, cold)}
    trends = {"hot": trend(hot, games([28])), "cold": trend(cold, games([8]))}
    results = strategy.suggest_opportunity_trades(players, [
        {"roster_id": 1, "players": ["hot"]}, {"roster_id": 2, "players": ["cold"]},
    ], 1, ("WR",), trends)
    assert results
    assert results[0]["give"][0]["trend"]["provisional"]
    assert trends["hot"]["perceived_value"] < trend(hot, games([28, 28, 28, 28]))["perceived_value"]
