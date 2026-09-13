from __future__ import annotations

import hashlib

import pytest

from sleeper_draft_assistant import web
from sleeper_draft_assistant.cache import cache_directory
from sleeper_draft_assistant.models import Player, Recommendation


@pytest.fixture
def configured_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "APP_PASSWORD_HASH", hashlib.sha256(b"correct horse").hexdigest()
    )
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-with-enough-entropy")


def test_signed_session_is_required_and_expires(configured_auth: None) -> None:
    token = web._session_token(timestamp=1_000_000)
    assert web._valid_session(token, timestamp=1_000_001)
    assert not web._valid_session(token + "tampered", timestamp=1_000_001)
    assert not web._valid_session(
        token, timestamp=1_000_000 + web.SESSION_SECONDS + 1
    )


def test_request_uses_mobile_defaults() -> None:
    options = web.RecommendationRequest()
    assert options.league == "shield-ai"
    assert options.candidates == 15
    assert options.runs_per_candidate == 10_000
    assert options.workers == "auto"


def test_request_caps_total_rollouts() -> None:
    with pytest.raises(ValueError, match="500,000"):
        web.RecommendationRequest(candidates=25, runs_per_candidate=25_000)


def test_recommendation_payload_contains_phone_metrics() -> None:
    player = Player(
        sleeper_id="1",
        name="Test Runner",
        position="RB",
        team="CHI",
        ecr=10,
        uncertainty=2,
    )
    recommendation = Recommendation(
        player=player,
        availability_rate=0.8,
        selection_rate=0.6,
        mean_score=1234.567,
        top_roster_rate=0.42,
        samples=10_000,
        starter_gain=22.2,
        next_pick_availability=0.1,
        wait_cost=12.4,
    )
    payload = web._recommendation_payload(1, recommendation)
    assert payload["name"] == "Test Runner"
    assert payload["model_score"] == 1234.57
    assert payload["samples"] == 10_000
    assert payload["next_turn_availability"] == 0.1


def test_cache_directory_can_be_configured(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SLEEPER_CACHE_DIR", str(tmp_path))
    assert cache_directory() == tmp_path.resolve()
