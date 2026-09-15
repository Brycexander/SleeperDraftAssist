"""Free, user-supplied expert snapshot loading.

No subscription or unverified scrape is assumed. A future provider adapter can
be added behind ``load_expert_board`` after its response shape is audited.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from collections import Counter
from typing import Any

from .expert_data import ExpertBoard, ExpertSelection, ExpertSnapshot, build_consensus, select_expert_panel


MAX_BYTES = 2 * 1024 * 1024


def _read_snapshot(path: Path) -> ExpertSnapshot:
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("expert snapshot exceeds the 2 MiB limit")
    return ExpertSnapshot.model_validate_json(path.read_bytes())


def _check_context(snapshot: ExpertSnapshot, *, season: int, week: int, scoring: str) -> None:
    if snapshot.season != season or snapshot.week != week or snapshot.scoring != scoring:
        raise ValueError("expert snapshot season, week, or scoring does not match the request")


def _selection_warnings(selection: ExpertSelection) -> list[str]:
    warnings = list(selection.warnings)
    counts = Counter(item.reason for item in selection.excluded)
    for reason, count in sorted(counts.items()):
        example = next(item.detail for item in selection.excluded if item.reason == reason)
        warnings.append(f"Excluded {count} expert(s) for {reason}: {example}")
    return warnings


def _excluded_rows(selection: ExpertSelection) -> list[dict[str, str]]:
    return [
        {"expert_id": item.expert_id, "reason": item.reason, "detail": item.detail}
        for item in selection.excluded
    ]


def _build_selected_board(selection: ExpertSelection) -> ExpertBoard:
    board = build_consensus(list(selection.selected))
    score_rows = tuple((item.expert_id, round(selection.scores[item.expert_id], 6)) for item in selection.selected)
    return ExpertBoard(
        values=board.values,
        panel_ids=board.panel_ids,
        universe_size=board.universe_size,
        warnings=board.warnings,
        panel_scores=score_rows,
        selection_notes=tuple(_selection_warnings(selection)),
    )


def import_expert_snapshot(raw: bytes, target_dir: Path, *, season: int, week: int,
                           scoring: str, now: datetime) -> dict[str, Any]:
    if len(raw) > MAX_BYTES:
        raise ValueError("expert snapshot exceeds the 2 MiB limit")
    snapshot = ExpertSnapshot.model_validate_json(raw)
    _check_context(snapshot, season=season, week=week, scoring=scoring)
    selection = select_expert_panel(
        snapshot.submissions, season=season, week=week, scoring=scoring, now=now
    )
    if not selection.selected:
        return {
            "status": "insufficient_data", "panel": 0, "universe": 0,
            "selection_method": "rolling_multi_year_ros_accuracy",
            "selected_experts": [], "excluded_experts": _excluded_rows(selection),
            "warnings": _selection_warnings(selection),
        }
    board = _build_selected_board(selection)
    target_dir.mkdir(parents=True, exist_ok=True)
    destination = target_dir / f"expert-ros-{season}-{week}-{scoring}.json"
    with tempfile.NamedTemporaryFile(mode="wb", dir=target_dir, prefix="expert-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "ready", "path": str(destination), "panel": len(board.panel_ids),
        "universe": len(board.values), "fetched_at": snapshot.fetched_at.isoformat(),
        "selection_method": "rolling_multi_year_ros_accuracy",
        "selected_experts": [
            {"expert_id": expert_id, "score": score} for expert_id, score in board.panel_scores
        ],
        "excluded_experts": _excluded_rows(selection),
    }


def load_expert_board(cache_dir: Path, *, season: int, week: int, scoring: str,
                      now: datetime, snapshot_path: Path | None = None,
                      refresh: bool = False) -> tuple[str, ExpertBoard | None, list[dict[str, Any]], list[str]]:
    """Load an imported, validated panel; no silent Sleeper/projection fallback."""
    path = snapshot_path or (cache_dir / f"expert-ros-{season}-{week}-{scoring}.json")
    try:
        snapshot = _read_snapshot(path)
        _check_context(snapshot, season=season, week=week, scoring=scoring)
        selection = select_expert_panel(
            snapshot.submissions, season=season, week=week, scoring=scoring, now=now
        )
        selection_warnings = _selection_warnings(selection)
        if not selection.selected:
            reasons = {item.reason for item in selection.excluded}
            status = "stale" if reasons and reasons <= {"stale_ranking", "future_publication"} else "insufficient_data"
            return status, None, [{
                "name": "Imported expert ROS snapshot", "url": "file import", "status": status,
                "selection_method": "rolling_multi_year_ros_accuracy",
                "selected_experts": [], "excluded_experts": _excluded_rows(selection),
            }], selection_warnings
        board = _build_selected_board(selection)
        if not board.values:
            return "insufficient_data", None, [], ["The expert snapshot has no players with sufficient panel coverage."]
        return "ready", board, [{
            "name": "Imported expert ROS snapshot", "url": "file import", "status": "loaded",
            "season": season, "week": week, "scoring": scoring, "panel": len(board.panel_ids),
            "selection_method": "rolling_multi_year_ros_accuracy",
            "selected_experts": [
                {"expert_id": expert_id, "score": score} for expert_id, score in board.panel_scores
            ],
            "excluded_experts": _excluded_rows(selection),
        }], list(board.warnings) + selection_warnings
    except FileNotFoundError:
        return "unavailable", None, [], ["No validated expert ROS snapshot is available. Import a free-source JSON snapshot before using expert transaction advice."]
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return "unavailable", None, [], [f"Expert ROS snapshot unavailable: {exc}"]
