from __future__ import annotations

import asyncio
import hashlib
import json
from types import SimpleNamespace

import pytest

from sleeper_draft_assistant.management import TeamService
from sleeper_draft_assistant.strategy import WeeklyPlayer
from sleeper_draft_assistant import web


class Client:
    def __init__(self):
        self.settings = {}
        self.team = {
            "league_id": "42", "name": "Example league", "season": "2026",
            "roster_positions": ["QB", "RB", "FLEX", "BN"],
            "scoring_settings": {"rec": 0.5, "pass_td": 6}, "settings": self.settings,
        }
        self.teams = [
            {"roster_id": 4, "owner_id": "owner", "players": ["qb", "rb", "wr", "ir"], "starters": ["qb", "rb", "0"], "reserve": ["ir"]},
            {"roster_id": 9, "owner_id": "other", "players": ["taken"], "starters": []},
        ]

    def nfl_state(self):
        return {"season": "2026", "week": 2, "season_type": "regular"}

    def user(self, username):
        return {"user_id": "owner"}

    def league(self, league_id):
        return self.team

    def rosters(self, league_id):
        return self.teams

    def matchups(self, league_id, week):
        return [{"roster_id": 4, "starters": ["qb", "rb", "wr"]}]


@pytest.fixture
def service(tmp_path):
    players = {
        key: WeeklyPlayer(key, key.upper(), position, "CHI", points, roster_value=points)
        for key, position, points in (
            ("qb", "QB", 20), ("rb", "RB", 10), ("wr", "WR", 12),
            ("ir", "WR", 30), ("taken", "RB", 25), ("free", "WR", 14),
        )
    }
    calls = []
    def loader(cache_dir, season, week, scoring, **kwargs):
        calls.append((season, week, scoring, kwargs))
        return SimpleNamespace(players=players, warnings=[], sources=[{"name": "Fixture", "season": season, "week": week}])
    result = TeamService(Client(), loader, tmp_path)
    result.calls = calls
    return result


def test_weekly_advice_uses_live_ownership_scoring_and_excludes_reserve(service):
    report = service.advise("42", "example")
    assert report["roster_id"] == 4
    assert report["lineup"]["projected_points"] == 42
    assert "ir" not in {p["player_id"] for p in report["lineup"]["starters"]}
    assert service.calls[0][:3] == (2026, 2, {"rec": 0.5, "pass_td": 6})
    assert report["week"] == 2


def test_waiver_open_slot_does_not_drop_and_excludes_opponents(service):
    report = service.advise("42", "example", action="waivers")
    assert report["suggestions"][0]["add"]["player_id"] == "free"
    assert report["suggestions"][0]["drop"] is None


def test_disabled_additions_and_trade_deadline(service):
    service.client.settings.update(disable_adds=1, trade_deadline=1)
    waivers = service.advise("42", "example", action="waivers")
    trades = service.advise("42", "example", action="trades")
    assert not waivers["suggestions"]
    assert not trades["suggestions"]
    assert any("disabled" in w for w in waivers["warnings"])
    assert any("deadline" in w for w in trades["warnings"])


def test_week_validation_prevents_historical_live_roster_mix(service):
    with pytest.raises(ValueError, match="past weeks"):
        service.advise("42", "example", week=1)
    with pytest.raises(ValueError, match="refresh the season"):
        service.advise("42", "example", expected_season=2025)
    with pytest.raises(ValueError, match="current game locks"):
        service.advise("42", "example", action="waivers", week=3)
    assert not service.calls


def test_unlabeled_tier_confirmation_scoped_to_explicit_season_week(service):
    with pytest.raises(ValueError, match="explicit season and week"):
        service.advise("42", "example", confirm_tiers_week=True)
    service.advise("42", "example", mode="tiers", week=2, expected_season=2026, confirm_tiers_week=True)
    assert service.calls[0][3]["confirm_tiers_week"] is True


def test_missing_player_is_explicit_and_cannot_be_recommended(service):
    service.client.teams[0]["players"].append("missing")
    report = service.advise("42", "example")
    assert any("missing from the player feed" in w for w in report["warnings"])
    assert "missing" not in {p["player_id"] for p in report["lineup"]["starters"]}


def test_renewal_requires_an_explicit_chain(service):
    old = dict(service.client.team, season="2025", league_id="old")
    renewed = dict(service.client.team, previous_league_id="old")
    service.client.league = lambda league_id: old
    service.client.user_leagues = lambda user_id, season: [renewed]
    report = service.advise("old", "example")
    assert report["league_id"] == "42"
    assert any("renewed" in w for w in report["warnings"])
    service.client.user_leagues = lambda user_id, season: []
    with pytest.raises(ValueError, match="choose its 2026 league ID"):
        service.advise("old", "example")


async def _http(path, payload=None, cookie=None):
    body = json.dumps(payload).encode() if payload is not None else b""
    headers = [(b"content-type", b"application/json")]
    if cookie:
        headers.append((b"cookie", f"{web.COOKIE_NAME}={cookie}".encode()))
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST" if payload is not None else "GET", "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": headers, "client": ("127.0.0.1", 1234), "server": ("test", 80)}
    sent = False
    async def receive():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        await asyncio.Event().wait()
    messages = []
    async def send(message):
        messages.append(message)
    await web.app(scope, receive, send)
    status = next(m["status"] for m in messages if m["type"] == "http.response.start")
    result = b"".join(m.get("body", b"") for m in messages)
    return status, result


def test_team_routes_require_auth_and_serve_working_advice(service, monkeypatch):
    monkeypatch.setenv("APP_PASSWORD_HASH", hashlib.sha256(b"example").hexdigest())
    monkeypatch.setenv("SESSION_SECRET", "fixture-session-secret")
    monkeypatch.setattr(web, "team_service", service)
    token = web._session_token()
    assert asyncio.run(_http("/api/team", {"league_id": "42"}))[0] == 401
    assert asyncio.run(_http("/api/team/state"))[0] == 401
    status, content = asyncio.run(_http("/team", cookie=token))
    assert status == 200 and b"Find my starting lineup" in content
    status, content = asyncio.run(_http("/api/team", {"league_id": "42"}, cookie=token))
    assert status == 200
    assert json.loads(content)["lineup"]["projected_points"] == 42
    status, content = asyncio.run(_http("/api/team", {"league_id": "42", "week": 1}, cookie=token))
    assert status == 400
    assert "past weeks" in json.loads(content)["detail"]
    assert asyncio.run(_http("/api/team", {"league_id": "../bad"}, cookie=token))[0] == 422


def test_weekly_cli_bypasses_draft_sync(monkeypatch, capsys):
    from sleeper_draft_assistant import cli
    monkeypatch.setattr(cli.SleeperClient, "sync", lambda *a: pytest.fail("Weekly advice must not sync a draft"))
    monkeypatch.setattr(cli, "_manage_team", lambda client, args: print(f"{args.command}:{args.week}:{args.mode}"))
    assert cli.main(["lineup", "--week", "2", "--mode", "tiers"]) == 0
    assert "lineup:2:tiers" in capsys.readouterr().out
