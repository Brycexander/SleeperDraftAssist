"""Scoped weekly inputs, with explicit freshness and missing-data semantics.

Only raw responses are cached. Scoring and tier confirmation are applied anew on
every call, so another league's scoring or a previous week's confirmation cannot
leak into a recommendation.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from typing import Any
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup
import requests

from .models import normalize_position
from .strategy import WeeklyPlayer


TIERS_URL = "https://fantasyfootballtiers.com/"
_TEAMS = frozenset("ARI ATL BAL BUF CAR CHI CIN CLE DAL DEN DET GB HOU IND JAX KC LAC LAR LV MIA MIN NE NO NYG NYJ PHI PIT SEA SF TB TEN WAS".split())
_POSITIONS = frozenset("QB RB WR TE K DEF DL DE DT LB DB CB S".split())


@dataclass(frozen=True, slots=True)
class WeeklyData:
    players: dict[str, WeeklyPlayer]
    warnings: list[str]
    sources: list[dict[str, Any]]
    season: int
    week: int


@dataclass(frozen=True, slots=True)
class _Fetched:
    data: Any
    fetched_at: str
    last_modified: str | None
    url: str
    cached: bool = False


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: Any) -> datetime | None:
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value / 1000 if value > 1e11 else value, timezone.utc)
        if not value:
            return None
        stamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def _fetch(
    cache_dir: Path, key: str, url: str, ttl: int, refresh: bool,
    *, params: dict[str, Any] | None = None, text: bool = False,
) -> _Fetched:
    # Include the URL and query in the cache identity; files never contain scored
    # players or the user's tier confirmation.
    identity = hashlib.sha256(json.dumps([url, params], sort_keys=True).encode()).hexdigest()[:12]
    target = cache_dir / f"weekly-{key}-{identity}.json"
    if not refresh:
        try:
            cached = json.loads(target.read_text())
            stamp = _timestamp(cached["fetched_at"])
            if stamp and 0 <= (_now() - stamp).total_seconds() < ttl:
                return _Fetched(cached["data"], cached["fetched_at"], cached.get("last_modified"), url, True)
        except (OSError, ValueError, KeyError, TypeError):
            pass
    response = requests.get(
        url, params=params, timeout=30,
    )
    response.raise_for_status()
    fetched = _Fetched(response.text if text else response.json(), _now().isoformat(), response.headers.get("Last-Modified"), url)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=cache_dir, prefix="weekly-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump({"data": fetched.data, "fetched_at": fetched.fetched_at, "last_modified": fetched.last_modified}, handle)
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return fetched


def _team(value: Any) -> str:
    value = str(value or "").upper()
    return {"LA": "LAR", "WSH": "WAS", "JAC": "JAX", "OAK": "LV", "SD": "LAC"}.get(value, value)


def _name(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[a-z0-9]+", ascii_name.lower())
    while tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv", "v"}:
        tokens.pop()
    return "".join(tokens)


def _stats(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value) for key, value in (row.get("stats") or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and not key.startswith(("adp_", "pos_adp_", "pts_"))
        and key != "gp"
    }


def score_weekly_stats(stats: dict[str, float], scoring: dict[str, float], position: str) -> float | None:
    """Score raw expected statistics without season heuristics or a zero floor."""
    if not stats:
        return None
    values = dict(stats)
    bonus = f"bonus_rec_{position.lower()}"
    if position in {"RB", "WR", "TE"} and bonus not in values and "rec" in values:
        values[bonus] = values["rec"]
    if not any(key in values for key, weight in scoring.items() if weight):
        return None
    return sum(value * scoring.get(key, 0.0) for key, value in values.items())


def _projection_rows(data: Any, season: int, week: int | None) -> tuple[dict[str, dict[str, Any]], int]:
    if not isinstance(data, list):
        raise ValueError("projection response is not a list")
    result: dict[str, dict[str, Any]] = {}
    rejected = 0
    for row in data:
        if not isinstance(row, dict):
            rejected += 1
            continue
        if (str(row.get("season")) != str(season)
                or row.get("week") != week
                or row.get("season_type") != "regular"
                or row.get("category", "proj") != "proj"):
            rejected += 1
            continue
        player_id = str(row.get("player_id") or "")
        if not player_id:
            continue
        old = result.get(player_id)
        stamp = _timestamp(row.get("updated_at") or row.get("last_modified"))
        old_stamp = _timestamp(old.get("updated_at") or old.get("last_modified")) if old else None
        if old is None or (stamp and (old_stamp is None or stamp > old_stamp)):
            result[player_id] = row
    return result, rejected


def _source(label: str, fetched: _Fetched, season: int | None, week: int | None, **extra: Any) -> dict[str, Any]:
    return {"name": label, "url": fetched.url, "fetched_at": fetched.fetched_at,
            "last_modified": fetched.last_modified, "cached": fetched.cached,
            "season": season, "week": week, **extra}


def _schedule_games(data: Any, week: int) -> list[dict[str, Any]]:
    if not isinstance(data, list):
        raise ValueError("schedule response is not a list")
    games = [row for row in data if isinstance(row, dict) and row.get("week") == week
             and _team(row.get("home")) in _TEAMS and _team(row.get("away")) in _TEAMS]
    teams = [_team(row[key]) for row in games for key in ("home", "away")]
    # A truncated schedule must not turn every absent team into a false bye.
    if len(games) < 10 or len(teams) != len(set(teams)):
        raise ValueError("schedule is incomplete or contains duplicate teams")
    return games


def _kickoffs(data: Any, season: int, week: int) -> dict[frozenset[str], tuple[datetime, bool]]:
    if not isinstance(data, dict):
        raise ValueError("kickoff response is not an object")
    if data.get("season", {}).get("year") != season or data.get("season", {}).get("type") != 2 or data.get("week", {}).get("number") != week:
        raise ValueError("kickoff response identifies a different season or week")
    result = {}
    for event in data.get("events", []):
        if event.get("season", {}).get("year") != season or event.get("season", {}).get("type") != 2 or event.get("week", {}).get("number") != week:
            continue
        stamp = _timestamp(event.get("date"))
        competitions = event.get("competitions") or []
        if not stamp or not competitions:
            continue
        competitors = competitions[0].get("competitors") or []
        teams = frozenset(_team(item.get("team", {}).get("abbreviation")) for item in competitors)
        if len(teams) == 2 and teams <= _TEAMS:
            started = event.get("status", {}).get("type", {}).get("state") in {"in", "post"}
            result[teams] = (stamp, started)
    return result


def _scope(html: str) -> tuple[int | None, int | None]:
    """Read explicit ranking labels, never incidental donation/copyright years."""
    soup = BeautifulSoup(html, "html.parser")
    season = week = None
    for node in soup.select("[data-season], [data-week], meta[name]"):
        key = str(node.get("name") or "").lower()
        year = node.get("data-season") or (node.get("content") if key in {"season", "nfl-season"} else None)
        number = node.get("data-week") or (node.get("content") if key in {"week", "nfl-week"} else None)
        if year and re.fullmatch(r"20\d{2}", str(year)):
            season = int(year)
        if number and re.fullmatch(r"\d{1,2}", str(number)):
            week = int(number)
    for node in soup.select("title, h1, h2, h3, h4, caption"):
        label = node.get_text(" ", strip=True)
        match = re.search(r"\bweek\s*(\d{1,2})\b", label, re.I)
        if match:
            week = int(match[1])
            match_year = re.search(r"\b(20\d{2})\b", label)
            if match_year:
                season = int(match_year[1])
    return season, week


def _tier_rows(html: str) -> list[tuple[int, int, str, str | None, str | None]]:
    """Return tier/list order/name/optional ID/team from the site's text lists."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    order = 0
    for item in soup.find_all("li"):
        match = re.match(r"\s*Tier\s+(\d+)\s*:\s*(.+)", item.get_text(" ", strip=True), re.I)
        if not match:
            continue
        tier = int(match[1])
        if tier < 1:
            continue
        for name in match[2].split(","):
            order += 1
            rows.append((tier, order, name.strip(), item.get("data-sleeper-id"), item.get("data-team")))
    return rows


def _tier_match(name: str, group: str, players: dict[str, WeeklyPlayer], player_id: str | None = None, team: str | None = None) -> str | None:
    eligible = {"RB", "WR", "TE"} if group == "FLEX" else {group}
    if player_id:
        candidate = players.get(str(player_id))
        return candidate.player_id if candidate and candidate.position in eligible else None
    normalized = _name(name)
    matches = [player.player_id for player in players.values()
               if player.position in eligible and player.rosterable
               and (not team or player.team == _team(team))
               and (_name(player.name) == normalized or (group == "DEF" and _name(player.team) == normalized))]
    return matches[0] if len(matches) == 1 else None


def _load_tiers(cache_dir: Path, season: int, week: int, scoring: dict[str, float], players: dict[str, WeeklyPlayer], refresh: bool, confirm_tiers_week: bool) -> tuple[dict[str, WeeklyPlayer], list[str], list[dict[str, Any]]]:
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []
    reception = scoring.get("rec", 0.0)
    scoring_type = {0.0: "STD", 0.5: "HALF", 1.0: "PPR"}.get(reception)
    if scoring_type is None:
        return players, ["Fantasy Football Tiers has no matching chart for this reception scoring; tier advice is unavailable."], sources
    try:
        home = _fetch(cache_dir, f"tiers-home-{season}-{week}", TIERS_URL, 900, refresh, text=True)
        source_season, source_week = _scope(home.data)
        if home.last_modified and (_now() - parsedate_to_datetime(home.last_modified)).total_seconds() > 7 * 86400:
            sources.append(_source("Fantasy Football Tiers", home, source_season, source_week, status="stale", scoring=scoring_type))
            return players, ["Fantasy Football Tiers was last modified more than seven days ago; its tiers were withheld."], sources
        if ((source_season is not None and source_season != season)
                or (source_week is not None and source_week != week)):
            sources.append(_source("Fantasy Football Tiers", home, source_season, source_week, status="rejected", scoring=scoring_type))
            return players, ["Fantasy Football Tiers identifies a different season or week; its tiers were withheld."], sources
        confirmed = source_season == season and source_week == week
        sources.append(_source("Fantasy Football Tiers", home, source_season, source_week, status="verified" if confirmed else "user_confirmed" if confirm_tiers_week else "unverified", scoring=scoring_type))
        if not confirmed and not confirm_tiers_week:
            return players, ["Fantasy Football Tiers does not label its season and week in machine-readable text. Tiers were withheld; check the linked charts and confirm they match the selected season/week to use them."], sources
        verification = "Source identifies season/week" if confirmed else "User-confirmed season/week; source does not label week"
        if not confirmed:
            warnings.append(f"Fantasy Football Tiers is user-confirmed for {season} week {week}; the source does not label its week in machine-readable text.")
        groups = [("QB", "QB"), ("RB", f"RB-{scoring_type}"), ("WR", f"WR-{scoring_type}"), ("TE", f"TE-{scoring_type}"), ("FLEX", f"FLX-{scoring_type}"), ("K", "K"), ("DEF", "DST")]
        result = dict(players)
        matched = unmatched = 0
        for group, filename in groups:
            url = f"{TIERS_URL}gallery_files/{filename}.html"
            try:
                fetched = _fetch(cache_dir, f"tiers-{filename}-{season}-{week}", url, 900, refresh, text=True)
                own_season, own_week = _scope(fetched.data)
                if ((own_season is not None and own_season != season) or (own_week is not None and own_week != week)):
                    raise ValueError("chart labels a different season or week")
                if fetched.last_modified:
                    modified = parsedate_to_datetime(fetched.last_modified)
                    if (_now() - modified).total_seconds() > 7 * 86400:
                        raise ValueError("chart was last modified more than seven days ago")
                rows = _tier_rows(fetched.data)
                if not rows:
                    raise ValueError("no tier lists found")
                sources.append(_source(f"Fantasy Football Tiers {group}", fetched, season, week, verification=verification, scoring=scoring_type, status="used"))
                for tier, rank, name, player_id, team in rows:
                    match = _tier_match(name, group, players, player_id, team)
                    if match is None:
                        unmatched += 1
                        continue
                    fields = {"flex_tier": tier, "flex_rank": float(rank)} if group == "FLEX" else {"tier": tier, "rank": float(rank)}
                    result[match] = replace(result[match], **fields, tier_source=f"{url} ({verification}; {season} week {week}; {scoring_type})")
                    matched += 1
            except (requests.RequestException, ValueError, TypeError, OSError) as error:
                warnings.append(f"Fantasy Football Tiers {group} unavailable: {error}")
        if unmatched:
            warnings.append(f"{unmatched} tier entries could not be matched unambiguously to an active Sleeper player and were skipped.")
        if matched:
            warnings.append("Tiers use the site's standard scoring chart; league-specific bonuses apply to projections only. Within-tier rank is chart list order; numerical upside is unavailable.")
        return result, warnings, sources
    except (requests.RequestException, ValueError, TypeError, OSError) as error:
        return players, [f"Fantasy Football Tiers unavailable: {error}"], sources


def load_expert_players(
    cache_dir: Path, season: int, week: int, scoring: dict[str, float],
    refresh: bool = False, confirm_tiers_week: bool = False,
) -> WeeklyData:
    """Load player identity and transaction locks without projection feeds."""
    if not 2009 <= season <= 2100 or not 1 <= week <= 18:
        raise ValueError("Choose an NFL season and regular-season week 1–18")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scoring.values()):
        raise ValueError("Scoring weights must be finite numbers")
    warnings = ["Expert transaction player eligibility uses Sleeper metadata and game locks; it does not load projection or tier feeds."]
    sources: list[dict[str, Any]] = []
    specs = {
        "players": ("Sleeper player metadata", "https://api.sleeper.app/v1/players/nfl", 86400, None),
        "schedule": ("Sleeper NFL schedule", f"https://api.sleeper.com/schedule/nfl/regular/{season}", 60, None),
        "kickoffs": ("ESPN kickoff times", "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard", 300, {"dates": season, "seasontype": 2, "week": week, "limit": 1000}),
    }
    fetched: dict[str, _Fetched] = {}
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            key: pool.submit(_fetch, cache_dir, f"{key}-{season}-{week}", url, ttl, refresh, params=params)
            for key, (_, url, ttl, params) in specs.items()
        }
        for key, future in futures.items():
            label, url, _, _ = specs[key]
            try:
                value = future.result()
                fetched[key] = value
                sources.append(_source(label, value, season, week, status="loaded"))
            except (requests.RequestException, ValueError, TypeError, OSError) as error:
                warnings.append(f"{label} unavailable: {error}")
                sources.append({"name": label, "url": url, "status": "unavailable", "season": season, "week": week})

    metadata = fetched["players"].data if "players" in fetched and isinstance(fetched["players"].data, dict) else {}
    games: list[dict[str, Any]] = []
    kickoffs: dict[frozenset[str], tuple[datetime, bool]] = {}
    try:
        if "schedule" in fetched:
            games = _schedule_games(fetched["schedule"].data, week)
    except (ValueError, TypeError, KeyError) as error:
        warnings.append(f"Sleeper NFL schedule unusable: {error}")
    try:
        if "kickoffs" in fetched:
            kickoffs = _kickoffs(fetched["kickoffs"].data, season, week)
    except (ValueError, TypeError, KeyError) as error:
        warnings.append(f"ESPN kickoff times unusable: {error}")

    team_games = {_team(row[key]): row for row in games for key in ("home", "away")}
    locks: dict[str, bool] = {}
    conservative = False
    now = _now()
    for team, game in team_games.items():
        pair = frozenset((_team(game["home"]), _team(game["away"])))
        kickoff = kickoffs.get(pair)
        status = str(game.get("status") or "").lower()
        already_started = status in {"in_progress", "in_game", "complete", "completed", "post_game", "final"}
        if kickoff:
            locks[team] = already_started or kickoff[1] or now >= kickoff[0]
        else:
            conservative = True
            game_date = str(game.get("date") or "")[:10]
            locks[team] = already_started or not game_date or game_date <= now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    if not games:
        warnings.append("NFL schedule unavailable or incomplete: all expert-mode roster moves are conservatively locked.")
    elif conservative:
        warnings.append("Some kickoff times are unavailable: affected expert-mode roster moves are conservatively locked.")

    players: dict[str, WeeklyPlayer] = {}
    for raw_id, raw in metadata.items():
        if not isinstance(raw, dict):
            continue
        player_id = str(raw_id)
        position = normalize_position(raw.get("position"))
        team = _team(raw.get("team"))
        name = raw.get("full_name") or " ".join(str(raw.get(key) or "") for key in ("first_name", "last_name")).strip() or player_id
        status = str(raw.get("status") or "")
        injury = str(raw.get("injury_status") or "")
        if injury and status.lower() in {"", "active"}:
            status = injury
        if not team:
            status = status if status.lower() in {"inactive", "retired"} else "No NFL team"
        eligible = tuple(normalize_position(value) for value in raw.get("fantasy_positions") or [position])
        players[player_id] = WeeklyPlayer(
            player_id=player_id, name=name, position=position, team=team,
            points=None, roster_value=0.0, status=status, eligible_positions=eligible,
            locked=locks.get(team, not bool(games)),
            bye=bool(team in _TEAMS and games and team not in team_games),
            rosterable=bool(team in _TEAMS and position in _POSITIONS and status.lower() not in {"retired", "inactive"}),
            source=f"Sleeper player metadata for {season} week {week}; expert ROS ranks supply long-term value",
        )
    return WeeklyData(players, warnings, sources, season, week)


def load_weekly_players(cache_dir: Path, season: int, week: int, scoring: dict[str, float], refresh: bool = False, confirm_tiers_week: bool = False) -> WeeklyData:
    """Load regular-season advice inputs; missing projections remain ``None``.

    ``roster_value`` is season projected points / 17, a season-strength proxy
    preserved during byes/injuries. It is not a weekly fallback, a true
    rest-of-season forecast, or a market trade value.
    """
    if not 2009 <= season <= 2100 or not 1 <= week <= 18:
        raise ValueError("Choose an NFL season and regular-season week 1–18")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scoring.values()):
        raise ValueError("Scoring weights must be finite numbers")
    warnings: list[str] = []
    sources: list[dict[str, Any]] = []
    specs = {
        "players": ("Sleeper player metadata", "https://api.sleeper.app/v1/players/nfl", 86400, None),
        "projections": ("Sleeper weekly projections", f"https://api.sleeper.com/projections/nfl/{season}/{week}", 900, {"season_type": "regular"}),
        "season": ("Sleeper season projection strength", f"https://api.sleeper.com/projections/nfl/{season}", 21600, {"season_type": "regular"}),
        "schedule": ("Sleeper NFL schedule", f"https://api.sleeper.com/schedule/nfl/regular/{season}", 60, None),
        "kickoffs": ("ESPN kickoff times", "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard", 300, {"dates": season, "seasontype": 2, "week": week, "limit": 1000}),
    }
    fetched: dict[str, _Fetched] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {key: pool.submit(_fetch, cache_dir, f"{key}-{season}-{week}", url, ttl, refresh, params=params)
                   for key, (_, url, ttl, params) in specs.items()}
        for key, future in futures.items():
            label, url, _, _ = specs[key]
            try:
                value = future.result()
                fetched[key] = value
                sources.append(_source(label, value, season, None if key == "season" else week, status="loaded"))
            except (requests.RequestException, ValueError, TypeError, OSError) as error:
                warnings.append(f"{label} unavailable: {error}")
                sources.append({"name": label, "url": url, "status": "unavailable", "season": season, "week": week})
    metadata = fetched["players"].data if "players" in fetched and isinstance(fetched["players"].data, dict) else {}
    projections: dict[str, dict[str, Any]] = {}
    season_rows: dict[str, dict[str, Any]] = {}
    for key, target_week in (("projections", week), ("season", None)):
        if key not in fetched:
            continue
        try:
            rows, rejected = _projection_rows(fetched[key].data, season, target_week)
            if rejected:
                warnings.append(f"Rejected {rejected} {key} rows with missing or mismatched season/week/type metadata.")
            if key == "projections":
                projections = rows
            else:
                season_rows = rows
        except (ValueError, TypeError) as error:
            warnings.append(f"Sleeper {key} unusable: {error}")
    games: list[dict[str, Any]] = []
    kickoffs: dict[frozenset[str], tuple[datetime, bool]] = {}
    for key in ("schedule", "kickoffs"):
        if key not in fetched:
            continue
        try:
            if key == "schedule":
                games = _schedule_games(fetched[key].data, week)
            else:
                kickoffs = _kickoffs(fetched[key].data, season, week)
        except (ValueError, TypeError, KeyError) as error:
            warnings.append(f"{specs[key][0]} unusable: {error}")
    team_games = {_team(row[key]): row for row in games for key in ("home", "away")}
    locks: dict[str, bool] = {}
    conservative = False
    now = _now()
    for team, game in team_games.items():
        pair = frozenset((_team(game["home"]), _team(game["away"])))
        kickoff = kickoffs.get(pair)
        status = str(game.get("status") or "").lower()
        already_started = status in {"in_progress", "in_game", "complete", "completed", "post_game", "final"}
        if kickoff:
            locks[team] = already_started or kickoff[1] or now >= kickoff[0]
        else:
            conservative = True
            # Dates alone cannot establish kickoff; keep today's games fixed.
            game_date = str(game.get("date") or "")[:10]
            locks[team] = already_started or not game_date or game_date <= now.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    if not games:
        warnings.append("NFL schedule unavailable or incomplete: all roster moves are conservatively locked because game locks and byes cannot be verified.")
    elif conservative:
        warnings.append("Some kickoff times are unavailable: players with games today or earlier are conservatively locked. Check Sleeper before adjusting them.")
    players: dict[str, WeeklyPlayer] = {}
    known_stats: set[str] = set()
    for player_id in set(metadata) | set(projections) | set(season_rows):
        base = metadata.get(player_id) or {}
        weekly = projections.get(player_id) or {}
        season_row = season_rows.get(player_id) or {}
        embedded = weekly.get("player") or season_row.get("player") or {}
        identity = {**base, **embedded}
        position = normalize_position(identity.get("position"))
        team = _team(weekly.get("team") or identity.get("team"))
        name = identity.get("full_name") or " ".join(str(identity.get(key) or "") for key in ("first_name", "last_name")).strip() or player_id
        status = str(base.get("status") or "")
        injury = str(identity.get("injury_status") or "")
        if injury and status.lower() in {"", "active"}:
            status = injury
        if not team:
            status = status if status.lower() in {"inactive", "retired"} else "No NFL team"
        stats = _stats(weekly)
        known_stats.update(stats)
        points = score_weekly_stats(stats, scoring, position)
        season_points = score_weekly_stats(_stats(season_row), scoring, position)
        bye = bool(team in _TEAMS and games and team not in team_games)
        if bye:
            points = 0.0
        eligible = tuple(normalize_position(value) for value in identity.get("fantasy_positions") or [position])
        players[player_id] = WeeklyPlayer(
            player_id=player_id, name=name, position=position, team=team,
            points=points, roster_value=max(0.0, season_points / 17.0) if season_points is not None else 0.0,
            status=status, eligible_positions=eligible,
            locked=locks.get(team, not bool(games)), bye=bye,
            rosterable=bool(team in _TEAMS and position in _POSITIONS and status.lower() not in {"retired", "inactive"}),
            source=f"No NFL game in {season} week {week}" if bye else f"Sleeper {season} week {week} raw projections; league scoring" if points is not None else "Weekly projection unavailable",
        )
    missing = sorted(key for key, value in scoring.items() if value and key not in known_stats and key not in {"bonus_rec_rb", "bonus_rec_wr", "bonus_rec_te"})
    if missing and projections:
        warnings.append("Weekly source does not project these scoring categories; their contribution is omitted: " + ", ".join(missing) + ".")
    if not any(player.points is not None and not player.bye for player in players.values()):
        warnings.append(f"No usable weekly projections for {season} week {week}; season projections have not been substituted.")
    if season_rows:
        warnings.append("Roster strength uses season projected points / 17 for injury/bye protection and trade balance. It is a season estimate, not a rest-of-season forecast or market trade value.")
    else:
        warnings.append("Season roster-strength values are unavailable; trade and drop protection cannot be fully evaluated.")
    players, tier_warnings, tier_sources = _load_tiers(cache_dir, season, week, scoring, players, refresh, confirm_tiers_week)
    warnings.extend(tier_warnings)
    sources.extend(tier_sources)
    return WeeklyData(players, warnings, sources, season, week)
