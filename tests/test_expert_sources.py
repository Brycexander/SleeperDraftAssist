from datetime import datetime, timedelta, timezone
import json
from sleeper_draft_assistant.expert_data import AccuracyEvidence, ExpertRow, ExpertSnapshot, ExpertSubmission
from sleeper_draft_assistant.expert_sources import import_expert_snapshot, load_expert_board


def payload():
    submissions = []
    for n in range(5):
        submissions.append(ExpertSubmission(expert_id=f"e{n}", name=f"E{n}", publisher=f"fixture-{n}",
            published_at=datetime.now(timezone.utc), source_url="https://example.invalid/ranks",
            accuracy_history=[
                AccuracyEvidence(season=2025, scoring="PPR", place=n + 1, field_size=100, graded_weeks=14, url="https://example.invalid/accuracy"),
                AccuracyEvidence(season=2024, scoring="PPR", place=n + 2, field_size=100, graded_weeks=14, url="https://example.invalid/accuracy"),
            ],
            rows=[ExpertRow(player_id=f"p{i}", position=("QB", "RB", "WR", "TE")[i % 4], overall_rank=i + 1) for i in range(100)]))
    return ExpertSnapshot(season=2026, week=2, scoring="PPR", fetched_at=datetime.now(timezone.utc), submissions=submissions).model_dump_json().encode()


def test_import_is_atomic_and_loads(tmp_path):
    raw = payload()
    result = import_expert_snapshot(raw, tmp_path, season=2026, week=2, scoring="PPR", now=datetime.now(timezone.utc))
    assert result["status"] == "ready"
    assert len(result["selected_experts"]) == 5
    assert result["selection_method"] == "rolling_multi_year_ros_accuracy"
    status, board, sources, warnings = load_expert_board(tmp_path, season=2026, week=2, scoring="PPR", now=datetime.now(timezone.utc))
    assert status == "ready" and board and len(board.panel_ids) == 5 and sources and not warnings
    assert len(board.panel_scores) == 5


def test_stale_snapshot_is_withheld(tmp_path):
    snapshot = json.loads(payload())
    old = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
    for submission in snapshot["submissions"]:
        submission["published_at"] = old
    path = tmp_path / "expert-ros-2026-2-PPR.json"
    path.write_text(json.dumps(snapshot))
    status, board, _, warnings = load_expert_board(tmp_path, season=2026, week=2, scoring="PPR", now=datetime.now(timezone.utc))
    assert status == "stale" and board is None and warnings


def test_legacy_single_season_accuracy_is_readable_but_not_qualified(tmp_path):
    snapshot = json.loads(payload())
    snapshot["schema_version"] = 1
    for submission in snapshot["submissions"]:
        submission["accuracy"] = submission.pop("accuracy_history")[0]
    path = tmp_path / "expert-ros-2026-2-PPR.json"
    path.write_text(json.dumps(snapshot))

    status, board, sources, warnings = load_expert_board(
        tmp_path, season=2026, week=2, scoring="PPR", now=datetime.now(timezone.utc)
    )

    assert status == "insufficient_data"
    assert board is None
    assert any("two of the last three" in warning for warning in warnings)
    assert len(sources[0]["excluded_experts"]) == 5
    assert all(item["reason"] == "insufficient_history" for item in sources[0]["excluded_experts"])


def test_import_returns_each_exclusion_when_too_few_experts_qualify(tmp_path):
    snapshot = json.loads(payload())
    for submission in snapshot["submissions"]:
        submission["accuracy"] = submission.pop("accuracy_history")[0]

    result = import_expert_snapshot(
        json.dumps(snapshot).encode(), tmp_path,
        season=2026, week=2, scoring="PPR", now=datetime.now(timezone.utc),
    )

    assert result["status"] == "insufficient_data"
    assert len(result["excluded_experts"]) == 5
    assert not (tmp_path / "expert-ros-2026-2-PPR.json").exists()
