from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from sleeper_draft_assistant.strategy import WeeklyPlayer
from sleeper_draft_assistant import weekly


NOW = datetime(2026, 9, 13, 15, tzinfo=timezone.utc)


def projection(player_id="one", week=1, season="2026", stats=None, position="WR"):
    return {"player_id": player_id, "week": week, "season": season,
            "season_type": "regular", "category": "proj", "team": "DAL",
            "player": {"first_name": "Test", "last_name": "Player", "position": position, "team": "DAL", "injury_status": None},
            "stats": {"rec": 5, "rec_yd": 60} if stats is None else stats}


@pytest.fixture
def feeds(monkeypatch):
    monkeypatch.setattr(weekly, "_now", lambda: NOW)
    teams = sorted(weekly._TEAMS)
    games = [{"home": teams[i], "away": teams[i + 1], "week": 1, "status": "pre_game", "date": "2026-09-13"}
             for i in range(0, len(teams), 2)]
    events = [{"date": "2026-09-13T17:00Z", "season": {"year": 2026, "type": 2}, "week": {"number": 1},
               "competitions": [{"competitors": [{"team": {"abbreviation": game[key]}} for key in ("home", "away")]}],
               "status": {"type": {"state": "pre"}}} for game in games]
    responses = {
        "players": {"one": {"full_name": "Test Player", "position": "WR", "team": "DAL", "status": "Active"},
                    "missing": {"full_name": "No Projection", "position": "RB", "team": "KC", "status": "Active"}},
        "projections": [projection()],
        "season": [projection(week=None, stats={"rec": 85, "rec_yd": 1020})],
        "schedule": games,
        "kickoffs": {"season": {"year": 2026, "type": 2}, "week": {"number": 1}, "events": events},
        "tiers": "<title>Fantasy Football Tiers</title><p>2025-2026 charity donations</p>",
    }
    def fetch(cache_dir, key, url, ttl, refresh, **kwargs):
        category = key.split("-")[0]
        data = responses[category]
        if isinstance(data, Exception):
            raise data
        return weekly._Fetched(data, NOW.isoformat(), "Sun, 13 Sep 2026 14:00:00 GMT", url)
    monkeypatch.setattr(weekly, "_fetch", fetch)
    return responses


def test_weekly_scoring_applies_custom_bonuses_once_and_keeps_negatives():
    assert weekly.score_weekly_stats({"rec": 4, "rec_yd": 50}, {"rec": 1, "rec_yd": .1, "bonus_rec_te": .5}, "TE") == 11
    assert weekly.score_weekly_stats({"rec": 4, "bonus_rec_te": 4}, {"rec": 1, "bonus_rec_te": .5}, "TE") == 6
    assert weekly.score_weekly_stats({"pts_allow_35p": 1}, {"pts_allow_35p": -4}, "DEF") == -4
    assert weekly.score_weekly_stats({"sack": 1}, {"sack": 1, "def_3_and_out": 1}, "DEF") == 1
    assert weekly.score_weekly_stats({}, {"rec": 1}, "WR") is None
    assert weekly.score_weekly_stats({"rec": 0}, {"rec": 1}, "WR") == 0


def test_loader_uses_actual_scoring_all_metadata_and_separate_season_values(tmp_path, feeds):
    ppr = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1, "rec_yd": .1})
    standard = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec_yd": .1})
    assert ppr.players["one"].points == 11
    assert ppr.players["one"].roster_value == 11
    assert standard.players["one"].points == 6
    assert "missing" in ppr.players
    assert ppr.players["missing"].points is None
    assert ppr.players["one"].tier is None
    assert any("does not label" in warning for warning in ppr.warnings)
    assert all(source.get("fetched_at") for source in ppr.sources)


def test_mismatched_projections_and_season_only_never_become_weekly_points(tmp_path, feeds):
    feeds["projections"] = [projection(week=None), projection(week=2), projection(season="2025")]
    result = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert result.players["one"].points is None
    assert result.players["one"].roster_value == 5
    assert any("Rejected 3" in warning for warning in result.warnings)
    assert any("not been substituted" in warning for warning in result.warnings)


def test_missing_raw_stats_do_not_use_preset_points_or_adp(tmp_path, feeds):
    feeds["projections"] = [projection(stats={"pts_ppr": 99, "adp_ppr": 1, "gp": 1})]
    result = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert result.players["one"].points is None


def test_kickoffs_keep_sunday_morning_players_unlocked_then_lock_at_start(tmp_path, feeds, monkeypatch):
    before = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert not before.players["one"].locked
    monkeypatch.setattr(weekly, "_now", lambda: datetime(2026, 9, 13, 17, tzinfo=timezone.utc))
    after = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert after.players["one"].locked


def test_injury_bye_and_missing_schedule_are_conservative(tmp_path, feeds):
    feeds["players"]["one"]["status"] = "Injured Reserve"
    feeds["projections"][0]["player"]["injury_status"] = "Out"
    result = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert result.players["one"].status == "Injured Reserve"
    assert result.players["one"].roster_value == 5
    feeds["schedule"] = [game for game in feeds["schedule"] if "DAL" not in (game["home"], game["away"])]
    result = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert result.players["one"].bye and result.players["one"].points == 0
    feeds["schedule"] = []
    result = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert result.players["one"].locked
    assert not result.players["one"].bye
    assert any("all roster moves" in warning for warning in result.warnings)


def test_missing_kickoffs_and_wrong_week_scoreboard_do_not_unlock_players(tmp_path, feeds):
    feeds["kickoffs"]["week"]["number"] = 2
    result = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert result.players["one"].locked
    assert any("kickoff times are unavailable" in warning for warning in result.warnings)


def test_failing_sources_return_partial_data_and_warnings(tmp_path, feeds):
    feeds["season"] = requests.ConnectionError("offline")
    feeds["projections"] = requests.ConnectionError("offline")
    result = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert result.players["one"].points is None
    assert result.players["one"].roster_value == 0
    assert any(source["status"] == "unavailable" for source in result.sources)


def test_tier_scope_ignores_donations_and_reads_explicit_labels():
    assert weekly._scope("<p>2025-2026 donations</p><h3>QB</h3>") == (None, None)
    assert weekly._scope("<h1>2026 Week 1 Rankings</h1>") == (2026, 1)
    assert weekly._scope('<meta name="season" content="2026"><meta name="week" content="2">') == (2026, 2)


def test_tier_names_are_normalized_but_ambiguous_matches_rejected():
    player = WeeklyPlayer("one", "Patrick Mahomes", "QB", "KC", 20)
    assert weekly._tier_match("Patrick Mahomes II", "QB", {"one": player}) == "one"
    other = replace(player, player_id="two", team="DAL")
    assert weekly._tier_match("Patrick Mahomes II", "QB", {"one": player, "two": other}) is None
    assert weekly._tier_match("Patrick Mahomes II", "QB", {"one": player, "two": other}, team="KC") == "one"
    assert weekly._tier_match("Patrick Mahomes", "WR", {"one": player}) is None


def test_tiers_require_current_confirmation_each_call_and_reject_wrong_scope(tmp_path, feeds):
    feeds["tiers"] = "<ul><li>Tier 1:&nbsp; Test Player</li></ul>"
    confirmed = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1}, confirm_tiers_week=True)
    assert confirmed.players["one"].tier == 1
    assert confirmed.players["one"].flex_tier == 1
    assert "User-confirmed" in confirmed.players["one"].tier_source
    unconfirmed = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1})
    assert unconfirmed.players["one"].tier is None
    feeds["tiers"] = "<h1>2025 Week 1</h1><li>Tier 1: Test Player</li>"
    wrong = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": 1}, confirm_tiers_week=True)
    assert wrong.players["one"].tier is None
    assert any("different season" in warning for warning in wrong.warnings)


def test_explicit_tier_scope_works_without_confirmation(tmp_path, feeds):
    feeds["tiers"] = "<h1>2026 Week 1</h1><li>Tier 2: Test Player</li>"
    result = weekly.load_weekly_players(tmp_path, 2026, 1, {"rec": .5})
    assert result.players["one"].tier == 2
    assert "HALF" in result.players["one"].tier_source


def test_confirmed_stale_tiers_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly, "_now", lambda: NOW)
    monkeypatch.setattr(weekly, "_fetch", lambda *args, **kwargs: weekly._Fetched("<li>Tier 1: Test Player</li>", NOW.isoformat(), "Tue, 01 Sep 2026 12:00:00 GMT", weekly.TIERS_URL))
    player = WeeklyPlayer("one", "Test Player", "WR", "DAL", 10)
    players, warnings, sources = weekly._load_tiers(tmp_path, 2026, 1, {"rec": 1}, {"one": player}, False, True)
    assert players["one"].tier is None
    assert sources[0]["status"] == "stale"
    assert any("seven days" in warning for warning in warnings)


def test_cache_is_scoped_and_expired_cache_does_not_hide_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(weekly, "_now", lambda: NOW)
    calls = []
    class Response:
        headers = {}
        def raise_for_status(self): pass
        def json(self): return {"raw": 1}
    def get(url, **kwargs):
        calls.append(url)
        return Response()
    monkeypatch.setattr(weekly.requests, "get", get)
    first = weekly._fetch(tmp_path, "proj-2026-1", "https://example.com/2026/1", 900, False)
    cached = weekly._fetch(tmp_path, "proj-2026-1", "https://example.com/2026/1", 900, False)
    different_week = weekly._fetch(tmp_path, "proj-2026-2", "https://example.com/2026/2", 900, False)
    assert not first.cached and cached.cached and not different_week.cached
    assert len(calls) == 2
    monkeypatch.setattr(weekly, "_now", lambda: datetime(2026, 9, 13, 16, tzinfo=timezone.utc))
    def failure(*args, **kwargs): raise requests.ConnectionError("offline")
    monkeypatch.setattr(weekly.requests, "get", failure)
    with pytest.raises(requests.ConnectionError):
        weekly._fetch(tmp_path, "proj-2026-1", "https://example.com/2026/1", 900, False)


@pytest.mark.parametrize("season,week", [(2026, 0), (2026, 19), (1900, 1)])
def test_invalid_week_rejected_without_fetch(tmp_path, season, week):
    with pytest.raises(ValueError):
        weekly.load_weekly_players(tmp_path, season, week, {"rec": 1})
