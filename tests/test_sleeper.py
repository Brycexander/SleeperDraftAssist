from __future__ import annotations

from sleeper_draft_assistant.sleeper import SleeperClient


def test_manager_biases_include_other_leagues_with_recency_and_shrinkage(
    monkeypatch,
) -> None:
    client = SleeperClient()
    leagues = {
        "main-2025": {"league_id": "main-2025", "previous_league_id": None},
        "other-2026": {"league_id": "other-2026", "previous_league_id": None},
    }
    drafts = {
        "main-2025": [{"draft_id": "draft-main", "status": "complete"}],
        "other-2026": [{"draft_id": "draft-other", "status": "complete"}],
    }
    picks = {
        "draft-main": [
            {
                "picked_by": "early-rb",
                "round": 1,
                "metadata": {"position": "RB"},
            },
            {
                "picked_by": "late-rb",
                "round": 5,
                "metadata": {"position": "RB"},
            },
        ],
        "draft-other": [
            {
                "picked_by": "early-rb",
                "round": 1,
                "metadata": {"position": "RB"},
            },
            {
                "picked_by": "late-rb",
                "round": 5,
                "metadata": {"position": "RB"},
            },
        ],
    }
    monkeypatch.setattr(client, "league", lambda league_id: leagues[league_id])
    monkeypatch.setattr(client, "drafts", lambda league_id: drafts[league_id])
    monkeypatch.setattr(client, "picks", lambda draft_id: picks[draft_id])
    current = {
        "league_id": "main-2026",
        "previous_league_id": "main-2025",
    }

    one_league = client.manager_position_biases(current, max_seasons=3)
    both_leagues = client.manager_position_biases(
        current,
        max_seasons=3,
        additional_league_ids=("other-2026",),
    )

    assert one_league["early-rb"]["RB"] < 0
    assert one_league["late-rb"]["RB"] > 0
    assert abs(one_league["early-rb"]["RB"]) < 1.0
    assert abs(both_leagues["early-rb"]["RB"]) > abs(
        one_league["early-rb"]["RB"]
    )
