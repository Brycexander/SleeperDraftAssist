from __future__ import annotations

from types import SimpleNamespace

from sleeper_draft_assistant import tier_charts
from sleeper_draft_assistant.strategy import WeeklyPlayer


def test_load_ppr_tier_charts_preserves_tiers_and_marks_active_roster(monkeypatch, tmp_path):
    pages = {
        "QB.html": "<li>Tier 1: Alpha Quarterback</li>",
        "RB-PPR.html": "<li>Tier 1: Alpha Runner, Free Runner</li>",
        "WR-PPR.html": "<li>Tier 2: Alpha Receiver</li>",
        "TE-PPR.html": "<li>Tier 1: Alpha Tight End</li>",
        "FLX-PPR.html": "<li>Tier 1: Alpha Runner, Alpha Receiver</li>",
    }

    def fetch(_cache_dir, _key, url, _ttl, _refresh, *, text=False, **_kwargs):
        filename = url.rsplit("/", 1)[-1]
        return SimpleNamespace(
            data=pages[filename],
            fetched_at="2026-09-14T12:00:00+00:00",
            last_modified=None,
            url=url,
            cached=False,
        )

    monkeypatch.setattr(tier_charts, "_fetch", fetch)
    players = {
        "qb": WeeklyPlayer("qb", "Alpha Quarterback", "QB", "CHI", 10),
        "rb": WeeklyPlayer("rb", "Alpha Runner", "RB", "CHI", 10),
        "wr": WeeklyPlayer("wr", "Alpha Receiver", "WR", "CHI", 10),
        "te": WeeklyPlayer("te", "Alpha Tight End", "TE", "CHI", 10),
        "free": WeeklyPlayer("free", "Free Runner", "RB", "DAL", 10),
    }

    charts, warnings, sources = tier_charts.load_ppr_tier_charts(
        tmp_path, season=2026, week=2, players=players, active_roster_ids={"qb", "rb", "wr"}
    )

    assert not warnings
    assert [chart["position"] for chart in charts] == ["QB", "RB", "WR", "TE", "FLEX"]
    runner = charts[1]["tiers"][0]["players"][0]
    free_runner = charts[1]["tiers"][0]["players"][1]
    assert runner["player_id"] == "rb" and runner["active_roster"] is True
    assert free_runner["player_id"] == "free" and free_runner["active_roster"] is False
    assert charts[4]["tiers"][0]["players"][0]["active_roster"] is True
    assert len(sources) == 5


def test_load_ppr_tier_charts_keeps_unmatched_names_visible(monkeypatch, tmp_path):
    def fetch(_cache_dir, _key, url, _ttl, _refresh, *, text=False, **_kwargs):
        return SimpleNamespace(
            data="<li>Tier 1: Unknown Player</li>",
            fetched_at="2026-09-14T12:00:00+00:00",
            last_modified=None,
            url=url,
            cached=False,
        )

    monkeypatch.setattr(tier_charts, "_fetch", fetch)
    charts, warnings, _sources = tier_charts.load_ppr_tier_charts(
        tmp_path, season=2026, week=2, players={}, active_roster_ids=set()
    )

    assert charts[0]["tiers"][0]["players"][0]["name"] == "Unknown Player"
    assert charts[0]["tiers"][0]["players"][0]["player_id"] is None
    assert any("could not be matched" in warning for warning in warnings)
