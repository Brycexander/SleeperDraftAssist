"""Recent offensive form from completed games, never live partial scores."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from .strategy import WeeklyPlayer, analyze_trade_trend
from .weekly import _fetch, _source, _stats, _team, _timestamp, score_weekly_stats


@dataclass(frozen=True)
class TradeHistory:
    trends: dict[str, dict[str, Any]]
    warnings: list[str]
    sources: list[dict[str, Any]]


def load_trade_history(cache_dir: Path, season: int, week: int, scoring: dict[str, float],
                       players: dict[str, WeeklyPlayer], refresh: bool = False) -> TradeHistory:
    """Inspect only the four weeks before the selected current week.

    Raw statistics use sparse zero-valued scoring categories. Games require an
    offensive snap and a completed matching schedule entry. No season crossover.
    """
    if not 1 <= week <= 18:
        raise ValueError("Trade history requires a regular-season week from 1 to 18")
    history: dict[str, list[dict[str, Any]]] = {pid: [] for pid in players}
    warnings, sources = [], []
    if week < 2:
        return TradeHistory({pid: analyze_trade_trend(p, []) for pid, p in players.items()},
                            ["No prior completed-week history yet. Buy-low/sell-high suggestions support one completed appearance, beginning in week 2; partial current-week results are excluded."], [])
    try:
        schedule = _fetch(cache_dir, f"schedule-{season}-{week}",
                          f"https://api.sleeper.com/schedule/nfl/regular/{season}", 60, refresh)
        if not isinstance(schedule.data, list):
            raise ValueError("Invalid schedule response")
        completed = {(g.get("week"), _team(g.get(side))) for g in schedule.data if isinstance(g, dict)
                     and str(g.get("status", "")).lower() in {"complete", "completed", "final", "post_game"}
                     for side in ("home", "away")}
        sources.append(_source("Sleeper completed-game verification", schedule, season, week, status="loaded"))
    except (requests.RequestException, ValueError, TypeError, OSError) as exc:
        return TradeHistory({pid: analyze_trade_trend(p, []) for pid, p in players.items()},
                            [f"Could not verify completed games; streak suggestions withheld: {exc}"], [])
    weeks = range(max(1, week - 4), week)
    with ThreadPoolExecutor(max_workers=4) as pool:
        tasks = {w: pool.submit(_fetch, cache_dir, f"trade-stats-{season}-{w}",
                               f"https://api.sleeper.com/stats/nfl/{season}/{w}", 3600, refresh,
                               params={"season_type": "regular"}) for w in weeks}
        for w, task in tasks.items():
            try:
                fetched = task.result()
                if not isinstance(fetched.data, list):
                    raise ValueError("Expected a list of player game stats")
                sources.append(_source("Sleeper completed-game stats", fetched, season, w, status="loaded"))
                rows = {}
                for row in fetched.data:
                    if not isinstance(row, dict) or str(row.get("season")) != str(season) or row.get("week") != w or row.get("category") != "stat" or row.get("season_type") != "regular":
                        continue
                    pid = str(row.get("player_id") or "")
                    if pid not in players or players[pid].position not in {"QB", "RB", "WR", "TE"}:
                        continue
                    old = rows.get(pid)
                    stamp = _timestamp(row.get("updated_at") or row.get("last_modified"))
                    old_stamp = _timestamp(old.get("updated_at") or old.get("last_modified")) if old else None
                    if old is None or (stamp and (old_stamp is None or stamp > old_stamp)):
                        rows[pid] = row
                for pid, row in rows.items():
                    stats = _stats(row)
                    team = _team(row.get("team"))
                    # Merely dressing for a game (gms_active) isn't an appearance.
                    if (w, team) not in completed or stats.get("off_snp", 0) <= 0:
                        continue
                    p = players[pid]
                    points = score_weekly_stats(stats, scoring, p.position)
                    if points is None:
                        points = 0.0  # Verified offensive appearance with no scoring events.
                    if p.position in {"WR", "TE"}:
                        work = stats.get("rec_tgt", 0)
                    elif p.position == "RB":
                        work = stats.get("rush_att", 0) + stats.get("rec_tgt", 0)
                    else:
                        work = stats.get("pass_att", 0) + stats.get("rush_att", 0)
                    snaps = stats.get("off_snp", 0)
                    team_snaps = stats.get("tm_off_snp", 0)
                    share = snaps / team_snaps if team_snaps > 0 and snaps <= team_snaps else None
                    history[pid].append({"week": w, "points": points, "opportunities": work,
                                         "snap_share": share, "team": team})
            except (requests.RequestException, ValueError, TypeError, OSError) as exc:
                warnings.append(f"Completed-game stats unavailable for week {w}: {exc}")
    trends = {pid: analyze_trade_trend(p, history[pid]) for pid, p in players.items()}
    if not any(t["signal"] in {"sell_high", "buy_low"} for t in trends.values()):
        warnings.append("No supported buy-low/sell-high signals in the available completed games. Missing history, changing roles, or ordinary scoring can explain this.")
    warnings.append("Streaks compare the last four completed weeks with a projection outlook. Missing appearances and byes are not scored as zero; the current week's games are excluded.")
    return TradeHistory(trends, warnings, sources)
