"""Recent actual performance compared with expert ROS ranks."""
from __future__ import annotations

from statistics import median
from typing import Any

from .expert_data import ExpertValue
from .strategy import WeeklyPlayer, _finite, _available


def add_positional_finishes(games: dict[str, list[dict[str, Any]]], players: dict[str, WeeklyPlayer]) -> dict[str, list[dict[str, Any]]]:
    by_week_position: dict[tuple[int, str], list[tuple[str, float]]] = {}
    for pid, rows in games.items():
        position = players.get(pid).position if pid in players else ""
        for row in rows:
            points = _finite(row.get("points"))
            if points is not None and position in {"QB", "RB", "WR", "TE"}:
                by_week_position.setdefault((int(row["week"]), position), []).append((pid, points))
    finishes: dict[tuple[int, str], dict[str, float]] = {}
    for key, values in by_week_position.items():
        ordered = sorted(values, key=lambda item: (-item[1], item[0]))
        for index, (pid, points) in enumerate(ordered):
            tied = [i + 1 for i, (_, value) in enumerate(ordered) if value == points]
            finishes.setdefault(key, {})[pid] = sum(tied) / len(tied)
    return {pid: [{**row, "position_finish": finishes.get((int(row["week"]), players[pid].position), {}).get(pid)} for row in rows] for pid, rows in games.items()}


def analyze_expert_form(player: WeeklyPlayer, value: ExpertValue | None, games: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in sorted(games, key=lambda row: int(row.get("week", 0)))[-4:] if _finite(row.get("position_finish")) is not None and _finite(row.get("points")) is not None]
    result = {"signal": "insufficient_data", "games": len(rows), "weeks": [row["week"] for row in rows],
              "recent_position_finish": None, "expected_position_rank": value.positional_rank if value else None,
              "recent_points": None, "workload_change": None, "snap_share_change": None,
              "confidence": "very_low", "provisional": len(rows) < 3,
              "reason": "At least one completed appearance and a qualified expert ROS rank are needed."}
    if value is None or not rows or not _available(player):
        if value is not None and not _available(player):
            result["signal"], result["reason"] = "injury_risk", "Current injury, bye, or eligibility status requires review before acting."
        return result
    if any(str(row.get("team", "")).upper() != player.team.upper() for row in rows if row.get("team")):
        return {**result, "signal": "role_change", "reason": "Historical team differs from the current team; old usage is not directly comparable."}
    if any(_finite(row.get("opportunities")) is None or _finite(row.get("opportunities")) <= 0 for row in rows):
        return {**result, "games": len(rows), "reason": "Offensive workload is missing or non-positive for part of the history."}
    expected = value.positional_rank
    recent_finish = median(float(row["position_finish"]) for row in rows)
    recent_points = sum(float(row["points"]) for row in rows) / len(rows)
    threshold = max(3.0, 0.25 * expected)
    confirmations = min(2, len(rows))
    hot = expected - recent_finish >= threshold and sum(float(row["position_finish"]) <= expected - threshold for row in rows[-3:]) >= confirmations
    cold = recent_finish - expected >= threshold and sum(float(row["position_finish"]) >= expected + threshold for row in rows[-3:]) >= confirmations
    comparison_size = min(2, len(rows) - 1) if len(rows) > 1 else 1
    earlier, latest = rows[:-comparison_size], rows[-comparison_size:]
    before = sum(float(row["opportunities"]) for row in earlier) / len(earlier) if earlier else None
    after = sum(float(row["opportunities"]) for row in latest) / len(latest)
    workload_change = after / before - 1 if before and before > 0 else None
    snap_change = None
    if earlier and all(_finite(row.get("snap_share")) is not None for row in rows):
        snap_change = sum(float(row["snap_share"]) for row in latest) / len(latest) - sum(float(row["snap_share"]) for row in earlier) / len(earlier)
    falling = workload_change is not None and workload_change < -.20 or snap_change is not None and snap_change < -.10
    rising = workload_change is not None and workload_change > .20 or snap_change is not None and snap_change > .10
    if falling:
        signal, reason = "role_decline", "Recent usage or snap share declined; the scoring slump may reflect a real role loss."
    elif hot and rising:
        signal, reason = "possible_breakout", "The player is outperforming the ROS rank with a larger role."
    elif hot:
        signal, reason = "sell_high", "Recent positional results are materially better than the expert ROS rank."
    elif cold:
        signal, reason = "buy_low", "Recent positional results are materially worse than the expert ROS rank while usage is stable."
    else:
        signal, reason = "neutral", "Recent positional results are not a material, consistent departure from the expert ROS rank."
    if len(rows) == 1 and signal in {"sell_high", "buy_low"}:
        reason = "Provisional one-game signal: recent positional result differs materially from the expert ROS rank; workload direction is unknown."
    return {**result, "signal": signal, "games": len(rows), "weeks": [row["week"] for row in rows],
            "recent_position_finish": round(recent_finish, 3), "recent_points": round(recent_points, 3),
            "workload_change": round(workload_change, 4) if workload_change is not None else None,
            "snap_share_change": round(snap_change, 4) if snap_change is not None else None,
            "confidence": "very_low" if len(rows) == 1 else "moderate" if len(rows) == 4 and snap_change is not None else "low",
            "provisional": len(rows) < 3, "reason": reason}
