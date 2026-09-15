from datetime import datetime, timezone
import requests

from sleeper_draft_assistant import weekly, strategy


def load(tmp_path, monkeypatch, week=5, altered=None):
    from sleeper_draft_assistant import trade_history
    def fetch(cache, key, url, ttl, refresh, **kwargs):
        if "/schedule/" in url:
            data = [{"week": w, "home": "CHI", "away": "DET", "status": "complete"} for w in range(1, 6)]
            if altered == "unfinished": data[2]["status"] = "in_progress"
        else:
            w = int(url.rsplit("/", 1)[1])
            if altered == "outage": raise requests.ConnectionError("offline")
            data = [{"player_id": "one", "team": "CHI", "season": "2026", "week": w,
                     "season_type": "regular", "category": "stat",
                     "stats": {"gp": 1, "rec": 2, "rec_yd": 30, "rec_tgt": 8, "off_snp": 50, "tm_off_snp": 60}}]
            if altered == "wrongseason": data[0]["season"] = "2025"
            if altered == "wrongweek": data[0]["week"] = 18
            if altered == "dnp": data[0]["stats"] = {"gms_active": 1}
            if altered == "duplicate": data.append(dict(data[0]))
        return weekly._Fetched(data, datetime.now(timezone.utc).isoformat(), None, url)
    monkeypatch.setattr(trade_history, "_fetch", fetch)
    p = strategy.WeeklyPlayer("one", "One", "WR", "CHI", 15, roster_value=15)
    return trade_history.load_trade_history(tmp_path, 2026, week, {"rec": 1, "rec_yd": .1}, {"one": p})


def test_recent_history_uses_league_scoring_and_four_prior_completed_weeks(tmp_path, monkeypatch):
    import importlib.util
    assert importlib.util.find_spec("sleeper_draft_assistant.trade_history"), "Historical form feed is not implemented"
    result = load(tmp_path, monkeypatch)
    assert result.trends["one"]["signal"] == "buy_low"
    assert result.trends["one"]["recent_points"] == 5
    assert result.trends["one"]["weeks"] == [1, 2, 3, 4]


def test_current_week_and_insufficient_early_season_are_not_streaks(tmp_path, monkeypatch):
    import importlib.util
    assert importlib.util.find_spec("sleeper_draft_assistant.trade_history")
    result = load(tmp_path, monkeypatch, week=1)
    assert result.trends["one"]["signal"] == "insufficient_data"
    assert result.warnings


def test_one_completed_week_is_available_in_week_two(tmp_path, monkeypatch):
    result = load(tmp_path, monkeypatch, week=2)
    assert result.trends["one"]["signal"] == "buy_low"
    assert result.trends["one"]["games"] == 1
    assert result.trends["one"]["provisional"]


def test_history_rejects_wrong_scope_dnp_and_handles_outage(tmp_path, monkeypatch):
    import importlib.util
    assert importlib.util.find_spec("sleeper_draft_assistant.trade_history")
    for altered in ("wrongseason", "wrongweek", "dnp", "outage"):
        result = load(tmp_path, monkeypatch, altered=altered)
        assert result.trends["one"]["signal"] == "insufficient_data"
        assert result.trends["one"]["games"] == 0


def test_history_deduplicates_and_excludes_incomplete_games(tmp_path, monkeypatch):
    import importlib.util
    assert importlib.util.find_spec("sleeper_draft_assistant.trade_history")
    result = load(tmp_path, monkeypatch, altered="duplicate")
    assert result.trends["one"]["games"] == 4
    result = load(tmp_path, monkeypatch, altered="unfinished")
    assert result.trends["one"]["weeks"] == [1, 2, 4]
