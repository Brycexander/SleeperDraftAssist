"""Fantasy Football Tiers PPR chart data prepared for the team UI."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from .strategy import WeeklyPlayer
from .weekly import TIERS_URL, _fetch, _tier_match, _tier_rows


_CHARTS: tuple[tuple[str, str, str], ...] = (
    ("QB", "QB.html", "Quarterback"),
    ("RB", "RB-PPR.html", "Running back"),
    ("WR", "WR-PPR.html", "Wide receiver"),
    ("TE", "TE-PPR.html", "Tight end"),
    ("FLEX", "FLX-PPR.html", "Flex"),
)


def _player_row(
    name: str,
    player_id: str | None,
    players: dict[str, WeeklyPlayer],
    active_roster_ids: set[str],
) -> dict[str, Any]:
    player = players.get(player_id) if player_id else None
    return {
        "name": name,
        "player_id": player_id,
        "position": player.position if player else None,
        "team": player.team if player else None,
        "active_roster": bool(player_id and player_id in active_roster_ids),
    }


def load_ppr_tier_charts(
    cache_dir: Path,
    season: int,
    week: int,
    players: dict[str, WeeklyPlayer],
    active_roster_ids: set[str],
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Load the site's PPR tier lists and mark players on the active roster.

    The source pages intentionally remain the authority for names and tier
    order. Unmatched names stay visible in the chart so a source/player-feed
    mismatch cannot silently remove a player from the user's view.
    """
    charts: list[dict[str, Any]] = []
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []
    unmatched = 0
    for position, filename, label in _CHARTS:
        url = f"{TIERS_URL}gallery_files/{filename}"
        try:
            fetched = _fetch(
                cache_dir,
                f"tiers-ppr-chart-{filename.removesuffix('.html')}-{season}-{week}",
                url,
                900,
                refresh,
                text=True,
            )
            rows = _tier_rows(fetched.data)
            if not rows:
                raise ValueError("no tier lists found")
            tier_map: dict[int, list[dict[str, Any]]] = {}
            for tier, _order, name, source_player_id, source_team in rows:
                player_id = _tier_match(
                    name,
                    position,
                    players,
                    source_player_id,
                    source_team,
                )
                if player_id is None:
                    unmatched += 1
                tier_map.setdefault(tier, []).append(
                    _player_row(name, player_id, players, active_roster_ids)
                )
            charts.append(
                {
                    "position": position,
                    "label": label,
                    "scoring": "PPR",
                    "source_url": url,
                    "tiers": [
                        {"tier": tier, "players": tier_map[tier]}
                        for tier in sorted(tier_map)
                    ],
                }
            )
            sources.append(
                {
                    "name": f"Fantasy Football Tiers {label} PPR",
                    "url": url,
                    "season": season,
                    "week": week,
                    "scoring": "PPR",
                    "fetched_at": fetched.fetched_at,
                    "last_modified": fetched.last_modified,
                    "cached": fetched.cached,
                    "status": "used",
                }
            )
        except (requests.RequestException, OSError, TypeError, ValueError) as error:
            warnings.append(f"Fantasy Football Tiers {label} PPR unavailable: {error}")
    if unmatched:
        warnings.append(
            f"{unmatched} PPR tier entries could not be matched to a Sleeper player; they remain visible without a roster highlight."
        )
    if not charts:
        warnings.append("No Fantasy Football Tiers PPR charts were available.")
    return charts, warnings, sources
