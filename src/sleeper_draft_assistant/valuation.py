from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
from datetime import date
import json
import math
from pathlib import Path
import re
from statistics import NormalDist
import time
import unicodedata
from typing import Any

from bs4 import BeautifulSoup
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import FLEX_POSITIONS, LeagueRules, Player, normalize_position


@dataclass(frozen=True, slots=True)
class RawProjection:
    name: str
    team: str
    position: str
    bye: int | None
    stats: dict[str, float]
    source: str = "FFToday"
    sleeper_id: str | None = None


@dataclass(frozen=True, slots=True)
class ValuationDiagnostics:
    projection_counts: dict[str, int]
    adp_counts: dict[str, int]
    replacement_points: dict[str, float]


@dataclass(frozen=True, slots=True)
class ValuationResult:
    players: tuple[Player, ...]
    diagnostics: ValuationDiagnostics


FFTODAY_POSITION_IDS = {
    "QB": 10,
    "RB": 20,
    "WR": 30,
    "TE": 40,
    "K": 80,
    "DEF": 99,
}

BASE_OUTCOME_CV = {
    "QB": 0.13,
    "RB": 0.22,
    "WR": 0.20,
    "TE": 0.23,
    "K": 0.18,
    "DEF": 0.25,
}


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "Chrome/127.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Referer": "https://www.fftoday.com/rankings/index.php",
        }
    )
    retries = Retry(
        total=3,
        backoff_factor=0.4,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    return session


def _number(value: str) -> float:
    cleaned = value.replace(",", "").replace("%", "").strip()
    if not cleaned or cleaned == "-":
        return 0.0
    return float(cleaned)


def _normalize_name(name: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    tokens = re.findall(r"[a-z0-9]+", ascii_name.lower())
    while tokens and tokens[-1] in {"jr", "sr", "ii", "iii", "iv"}:
        tokens.pop()
    return "".join(tokens)


def _projection_cache(cache_dir: Path, season: int) -> Path:
    return cache_dir / f"fftoday-projections-{season}-{date.today().isoformat()}.json"


def load_fftoday_projections(
    cache_dir: Path, season: int, refresh: bool = False
) -> list[RawProjection]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _projection_cache(cache_dir, season)
    if cache_file.exists() and not refresh:
        return [RawProjection(**row) for row in json.loads(cache_file.read_text())]

    session = _session()
    projections: list[RawProjection] = []
    for position, position_id in FFTODAY_POSITION_IDS.items():
        position_cache = cache_dir / (
            f"fftoday-projections-{season}-{date.today().isoformat()}-{position}.json"
        )
        if position_cache.exists() and not refresh:
            projections.extend(
                RawProjection(**row) for row in json.loads(position_cache.read_text())
            )
            continue

        position_projections: list[RawProjection] = []
        seen: set[str] = set()
        for page in range(5):
            try:
                response = session.get(
                    "https://www.fftoday.com/rankings/playerproj.php",
                    params={
                        "Season": season,
                        "PosID": position_id,
                        "order_by": "FFPts",
                        "sort_order": "DESC",
                        "cur_page": page,
                    },
                    timeout=25,
                )
                if response.status_code == 403:
                    time.sleep(2.0)
                    session = _session()
                    response = session.get(response.url, timeout=25)
                response.raise_for_status()
            except requests.RequestException:
                previous = sorted(
                    cache_dir.glob(f"fftoday-projections-{season}-*.json"),
                    reverse=True,
                )
                if previous:
                    return [RawProjection(**row) for row in json.loads(previous[0].read_text())]
                raise
            page_rows = _parse_fftoday_page(response.text, position)
            new_rows = [row for row in page_rows if _normalize_name(row.name) not in seen]
            if not new_rows:
                break
            position_projections.extend(new_rows)
            seen.update(_normalize_name(row.name) for row in new_rows)
            time.sleep(0.8)
            if len(page_rows) < 50:
                break
        projections.extend(position_projections)
        position_cache.write_text(
            json.dumps([asdict(row) for row in position_projections], indent=2)
        )

    cache_file.write_text(json.dumps([asdict(row) for row in projections], indent=2))
    return projections


def _parse_fftoday_page(html: str, position: str) -> list[RawProjection]:
    soup = BeautifulSoup(html, "html.parser")
    candidate_tables = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr", recursive=False)
        if len(rows) < 3:
            continue
        header = " ".join(cell.get_text(" ", strip=True) for cell in rows[1].find_all("td", recursive=False))
        if "FPts" in header and ("Player" in header or "Team" in header):
            candidate_tables.append((table, rows))
    if not candidate_tables:
        raise ValueError(f"Could not find FFToday {position} projection table")

    _, rows = candidate_tables[-1]
    projections = []
    for row in rows[2:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td", recursive=False)]
        parsed = _parse_fftoday_row(position, cells)
        if parsed is not None:
            projections.append(parsed)
    return projections


def _parse_fftoday_row(position: str, cells: list[str]) -> RawProjection | None:
    minimum_cells = {"QB": 13, "RB": 11, "WR": 11, "TE": 8, "K": 10, "DEF": 12}
    if len(cells) < minimum_cells[position] or not cells[1]:
        return None

    name = cells[1]
    if position == "DEF":
        team = ""
        bye = int(_number(cells[2])) or None
        stats = {
            "sack": _number(cells[3]),
            "fum_rec": _number(cells[4]),
            "int": _number(cells[5]),
            "def_td": _number(cells[6]),
            "pts_allow": _number(cells[7]),
            "pass_yd_allow_pg": _number(cells[8]),
            "rush_yd_allow_pg": _number(cells[9]),
            "safe": _number(cells[10]),
            "st_td": _number(cells[11]),
        }
    else:
        team = cells[2]
        bye = int(_number(cells[3])) or None
        values = [_number(value) for value in cells[4:]]
        if position == "QB":
            keys = (
                "pass_cmp",
                "pass_att",
                "pass_yd",
                "pass_td",
                "pass_int",
                "rush_att",
                "rush_yd",
                "rush_td",
                "source_fpts",
            )
        elif position == "RB":
            keys = (
                "rush_att",
                "rush_yd",
                "rush_td",
                "rec",
                "rec_yd",
                "rec_td",
                "source_fpts",
            )
        elif position == "WR":
            keys = (
                "rec",
                "rec_yd",
                "rec_td",
                "rush_att",
                "rush_yd",
                "rush_td",
                "source_fpts",
            )
        elif position == "TE":
            keys = ("rec", "rec_yd", "rec_td", "source_fpts")
        else:
            keys = ("fgm", "fga", "fg_pct", "xpm", "xpa", "source_fpts")
        stats = dict(zip(keys, values, strict=False))

    return RawProjection(
        name=name,
        team=team,
        position=position,
        bye=bye,
        stats=stats,
    )


def load_sleeper_metadata(cache_dir: Path, refresh: bool = False) -> dict[str, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"sleeper-players-{date.today().isoformat()}.json"
    if cache_file.exists() and not refresh:
        return json.loads(cache_file.read_text())
    try:
        response = _session().get(
            "https://api.sleeper.app/v1/players/nfl",
            params={"active": "true"},
            timeout=40,
        )
        response.raise_for_status()
    except requests.RequestException:
        previous = sorted(cache_dir.glob("sleeper-players-*.json"), reverse=True)
        if previous:
            return json.loads(previous[0].read_text())
        raise
    players = response.json()
    cache_file.write_text(json.dumps(players))
    return players


def load_sleeper_projections(
    cache_dir: Path, season: int, refresh: bool = False
) -> tuple[list[RawProjection], dict[str, float]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"sleeper-projections-{season}-{date.today().isoformat()}.json"
    if cache_file.exists() and not refresh:
        payload = json.loads(cache_file.read_text())
        return (
            [RawProjection(**row) for row in payload["projections"]],
            {key: float(value) for key, value in payload["adp"].items()},
        )

    response = _session().get(
        f"https://api.sleeper.com/projections/nfl/{season}",
        params={"season_type": "regular"},
        timeout=45,
    )
    response.raise_for_status()
    projections: list[RawProjection] = []
    adp: dict[str, float] = {}
    for row in response.json():
        player_data = row.get("player") or {}
        position = normalize_position(player_data.get("position"))
        if position not in FFTODAY_POSITION_IDS:
            continue
        player_id = str(row.get("player_id") or "")
        if not player_id:
            continue
        raw_stats = row.get("stats") or {}
        stats = {
            key: float(value)
            for key, value in raw_stats.items()
            if isinstance(value, (int, float))
        }
        market_adp = stats.get("adp_ppr")
        if market_adp is not None and 0 < market_adp < 999:
            adp[player_id] = market_adp

        scoring_stats = {
            key: value
            for key, value in stats.items()
            if not key.startswith(("adp_", "pts_")) and key != "gp"
        }
        if not scoring_stats:
            continue
        name = " ".join(
            part
            for part in (player_data.get("first_name"), player_data.get("last_name"))
            if part
        )
        projections.append(
            RawProjection(
                name=name or str(row.get("team") or player_id),
                team=str(row.get("team") or player_data.get("team") or ""),
                position=position,
                bye=None,
                stats=scoring_stats,
                source="Sleeper",
                sleeper_id=player_id,
            )
        )

    cache_file.write_text(
        json.dumps(
            {
                "projections": [asdict(projection) for projection in projections],
                "adp": adp,
            },
            indent=2,
        )
    )
    return projections, adp


def load_public_sleeper_adp(cache_dir: Path, refresh: bool = False) -> dict[str, float]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"sleeper-adp-{date.today().isoformat()}.json"
    if cache_file.exists() and not refresh:
        return {key: float(value) for key, value in json.loads(cache_file.read_text()).items()}

    try:
        response = _session().get(
            "https://www.fantasypros.com/nfl/adp/ppr-overall.php",
            timeout=25,
        )
        response.raise_for_status()
    except requests.RequestException:
        previous = sorted(cache_dir.glob("sleeper-adp-*.json"), reverse=True)
        if previous:
            return {
                key: float(value) for key, value in json.loads(previous[0].read_text()).items()
            }
        return {}
    soup = BeautifulSoup(response.text, "html.parser")
    config = None
    marker = "window.FP.reportConfig ="
    decoder = json.JSONDecoder()
    for script in soup.find_all("script"):
        text = script.string or script.get_text()
        index = text.find(marker)
        if index >= 0:
            payload = text[index + len(marker) :].lstrip()
            config, _ = decoder.raw_decode(payload)
            break
    if config is None:
        return {}

    sleeper_key = next(
        (
            field["key"]
            for field in config.get("table", {}).get("fields", [])
            if field.get("label") == "Sleeper"
        ),
        None,
    )
    adp = {
        _normalize_name(row["player"]["name"]): float(row[sleeper_key])
        for row in config.get("table", {}).get("rows", [])
        if sleeper_key and row.get(sleeper_key) is not None
    }
    cache_file.write_text(json.dumps(adp, indent=2))
    return adp


def _injury_and_role_risk(metadata: dict[str, Any], position: str) -> tuple[float, float, float]:
    injury_status = str(metadata.get("injury_status") or "").lower()
    status = str(metadata.get("status") or "").lower()
    injury_risk = {
        "out": 0.60,
        "doubtful": 0.45,
        "questionable": 0.22,
        "probable": 0.08,
    }.get(injury_status, 0.0)
    mean_multiplier = 1.0
    if any(label in status for label in ("injured reserve", "pup", "inactive")):
        injury_risk = max(injury_risk, 0.75)
        mean_multiplier = 0.82
    elif "suspend" in status:
        injury_risk = max(injury_risk, 0.55)
        mean_multiplier = 0.88
    elif injury_status == "out":
        mean_multiplier = 0.94

    depth_order = metadata.get("depth_chart_order")
    if isinstance(depth_order, (int, float)) and depth_order > 0:
        role_risk = min(0.75, 0.05 + 0.20 * (float(depth_order) - 1.0))
    else:
        role_risk = 0.28
    if not metadata.get("team"):
        role_risk = max(role_risk, 0.80)
        mean_multiplier *= 0.75
    if int(metadata.get("years_exp") or 0) == 0:
        role_risk = min(1.0, role_risk + 0.12)
    age = int(metadata.get("age") or 0)
    age_threshold = {"RB": 29, "WR": 31, "TE": 32, "QB": 36}.get(position, 99)
    if age >= age_threshold:
        role_risk = min(1.0, role_risk + 0.12)
    return injury_risk, role_risk, mean_multiplier


def _expected_bucket_score(
    mean: float, standard_deviation: float, buckets: list[tuple[float, float, float]]
) -> float:
    normal = NormalDist(mu=mean, sigma=standard_deviation)
    expected = 0.0
    for lower, upper, score in buckets:
        probability = normal.cdf(upper) - normal.cdf(lower)
        expected += probability * score
    return expected


def score_projection(projection: RawProjection, scoring: dict[str, float]) -> float:
    stats = projection.stats
    position = projection.position
    if projection.source == "Sleeper":
        points = sum(value * scoring.get(stat, 0.0) for stat, value in stats.items())
        if position == "DEF" and "def_3_and_out" not in stats:
            estimated_three_and_outs = max(
                42.5, 54.4 + (stats.get("sack", 42.5) - 42.5) * 0.30
            )
            points += estimated_three_and_outs * scoring.get("def_3_and_out", 0.0)
        return max(0.0, points)

    points = 0.0
    for stat in (
        "pass_cmp",
        "pass_att",
        "pass_yd",
        "pass_td",
        "pass_int",
        "rush_att",
        "rush_yd",
        "rush_td",
        "rec",
        "rec_yd",
        "rec_td",
    ):
        points += stats.get(stat, 0.0) * scoring.get(stat, 0.0)

    if position in {"RB", "WR", "TE"}:
        points += stats.get("rec", 0.0) * scoring.get(f"bonus_rec_{position.lower()}", 0.0)

    touches = stats.get("rush_att", 0.0) + stats.get("rec", 0.0)
    if position == "QB":
        estimated_fumbles = stats.get("pass_att", 0.0) * 0.012 + touches * 0.008
    else:
        estimated_fumbles = touches * 0.006
    estimated_lost = estimated_fumbles * 0.5
    points += estimated_fumbles * scoring.get("fum", 0.0)
    points += estimated_lost * scoring.get("fum_lost", 0.0)

    if position == "K":
        fgm = stats.get("fgm", 0.0)
        fga = stats.get("fga", fgm)
        buckets = (
            (0.13, "fgm_0_19"),
            (0.17, "fgm_20_29"),
            (0.27, "fgm_30_39"),
            (0.25, "fgm_40_49"),
            (0.18, "fgm_50p"),
        )
        if any(scoring.get(key, 0.0) for _, key in buckets):
            points += fgm * sum(weight * scoring.get(key, 0.0) for weight, key in buckets)
        else:
            points += fgm * scoring.get("fgm", 0.0)
        points += max(0.0, fga - fgm) * scoring.get("fgmiss", 0.0)
        points += stats.get("xpm", 0.0) * scoring.get("xpm", 0.0)
        points += max(0.0, stats.get("xpa", 0.0) - stats.get("xpm", 0.0)) * scoring.get(
            "xpmiss", 0.0
        )

    if position == "DEF":
        points += stats.get("sack", 0.0) * scoring.get("sack", 0.0)
        points += stats.get("int", 0.0) * scoring.get("int", 0.0)
        points += stats.get("fum_rec", 0.0) * scoring.get("fum_rec", 0.0)
        points += stats.get("fum_rec", 0.0) * 1.25 * scoring.get("ff", 0.0)
        points += stats.get("def_td", 0.0) * scoring.get("def_td", 0.0)
        points += stats.get("st_td", 0.0) * scoring.get("st_td", 0.0)
        points += stats.get("safe", 0.0) * scoring.get("safe", 0.0)

        games = 17.0
        average_points = stats.get("pts_allow", 0.0) / games
        point_buckets = [
            (-math.inf, 0.5, scoring.get("pts_allow_0", 0.0)),
            (0.5, 6.5, scoring.get("pts_allow_1_6", 0.0)),
            (6.5, 13.5, scoring.get("pts_allow_7_13", 0.0)),
            (13.5, 20.5, scoring.get("pts_allow_14_20", 0.0)),
            (20.5, 27.5, scoring.get("pts_allow_21_27", 0.0)),
            (27.5, 34.5, scoring.get("pts_allow_28_34", 0.0)),
            (34.5, math.inf, scoring.get("pts_allow_35p", 0.0)),
        ]
        points += games * _expected_bucket_score(average_points, 7.5, point_buckets)

        average_yards = stats.get("pass_yd_allow_pg", 0.0) + stats.get(
            "rush_yd_allow_pg", 0.0
        )
        yard_buckets = [
            (-math.inf, 100.5, scoring.get("yds_allow_0_100", 0.0)),
            (100.5, 199.5, scoring.get("yds_allow_100_199", 0.0)),
            (199.5, 299.5, scoring.get("yds_allow_200_299", 0.0)),
            (299.5, 349.5, scoring.get("yds_allow_300_349", 0.0)),
            (349.5, 399.5, scoring.get("yds_allow_350_399", 0.0)),
            (399.5, 449.5, scoring.get("yds_allow_400_449", 0.0)),
            (449.5, 499.5, scoring.get("yds_allow_450_499", 0.0)),
            (499.5, 549.5, scoring.get("yds_allow_500_549", 0.0)),
            (549.5, math.inf, scoring.get("yds_allow_550p", 0.0)),
        ]
        points += games * _expected_bucket_score(average_yards, 72.0, yard_buckets)
        estimated_three_and_outs = games * max(
            2.5, 3.2 + (stats.get("sack", 0.0) / games - 2.5) * 0.30
        )
        points += estimated_three_and_outs * scoring.get("def_3_and_out", 0.0)
    return max(0.0, points)


def _impute_projection_points(player: Player, known: list[tuple[float, float]]) -> float:
    if not known:
        return max(1.0, 260.0 * math.exp(-player.ecr / 100.0))
    known.sort()
    ranks = np.array([rank for rank, _ in known], dtype=float)
    points = np.array([value for _, value in known], dtype=float)
    if player.ecr <= ranks[-1]:
        return float(np.interp(player.ecr, ranks, points))
    return max(1.0, float(points[-1]) * math.exp(-(player.ecr - ranks[-1]) / 90.0))


def build_player_values(
    players: list[Player],
    rules: LeagueRules,
    cache_dir: Path,
    season: int,
    refresh: bool = False,
) -> ValuationResult:
    try:
        projections, sleeper_adp = load_sleeper_projections(
            cache_dir, season, refresh
        )
    except requests.RequestException:
        projections = load_fftoday_projections(cache_dir, season, refresh)
        sleeper_adp = {}
    sleeper = load_sleeper_metadata(cache_dir, refresh)
    public_adp = load_public_sleeper_adp(cache_dir, refresh)
    projection_map = {
        (_normalize_name(projection.name), projection.position): projection
        for projection in projections
    }
    projection_id_map = {
        projection.sleeper_id: projection
        for projection in projections
        if projection.sleeper_id is not None
    }

    scored: list[Player] = []
    known_by_position: dict[str, list[tuple[float, float]]] = {
        position: [] for position in FFTODAY_POSITION_IDS
    }
    source_counts = Counter[str]()
    for player in players:
        projection = projection_id_map.get(player.sleeper_id) or projection_map.get(
            (_normalize_name(player.name), player.position)
        )
        metadata = sleeper.get(player.sleeper_id, {})
        injury_risk, role_risk, mean_multiplier = _injury_and_role_risk(
            metadata, player.position
        )
        if projection is not None:
            projected_points = score_projection(projection, rules.scoring) * mean_multiplier
            source = projection.source
            known_by_position[player.position].append((player.ecr, projected_points))
        else:
            projected_points = 0.0
            source = "ECR-imputed"
        scored.append(
            replace(
                player,
                bye=projection.bye if projection and projection.bye else player.bye,
                projected_points=projected_points,
                injury_risk=injury_risk,
                role_risk=role_risk,
                projection_source=source,
            )
        )
        source_counts[source] += 1

    completed: list[Player] = []
    for player in scored:
        if player.projected_points > 0:
            completed.append(player)
            continue
        completed.append(
            replace(
                player,
                projected_points=_impute_projection_points(
                    player, known_by_position.get(player.position, [])
                ),
            )
        )

    adp_counts = Counter[str]()
    with_adp = []
    for player in completed:
        name_key = _normalize_name(player.name)
        if player.sleeper_id in sleeper_adp:
            adp = sleeper_adp[player.sleeper_id]
            adp_source = "Sleeper projection ADP"
        elif name_key in public_adp:
            adp = public_adp[name_key]
            adp_source = "FantasyPros Sleeper ADP"
        else:
            adp = player.ecr
            adp_source = "ECR fallback"
        adp_counts[adp_source] += 1
        with_adp.append(
            replace(
                player,
                adp=float(adp),
                adp_uncertainty=max(1.5, 0.07 * adp + 0.30 * abs(adp - player.ecr)),
            )
        )

    replacement_demand = {
        "QB": rules.teams * rules.required_count("QB"),
        "RB": rules.teams * rules.required_count("RB"),
        "WR": rules.teams * rules.required_count("WR"),
        "TE": rules.teams * rules.required_count("TE"),
        "K": rules.teams * rules.required_count("K"),
        "DEF": rules.teams * rules.required_count("DEF"),
    }
    points_by_position = {
        position: sorted(
            (player.projected_points for player in with_adp if player.position == position),
            reverse=True,
        )
        for position in replacement_demand
    }
    for _ in range(rules.teams * rules.flex_slots):
        flex_candidates = {
            position: points[replacement_demand[position]]
            for position in FLEX_POSITIONS
            if (points := points_by_position.get(position, []))
            and replacement_demand[position] < len(points)
        }
        if not flex_candidates:
            break
        selected_position = max(flex_candidates, key=flex_candidates.get)
        replacement_demand[selected_position] += 1

    replacements = {}
    for position, demand in replacement_demand.items():
        position_points = points_by_position[position]
        if not position_points:
            replacements[position] = 0.0
            continue
        index = min(len(position_points) - 1, max(0, demand))
        replacements[position] = position_points[index]

    with_vorp = []
    for player in with_adp:
        vorp = player.projected_points - replacements.get(player.position, 0.0)
        disagreement = min(0.12, player.uncertainty / max(10.0, player.ecr + 10.0) * 0.35)
        outcome_cv = min(
            0.60,
            BASE_OUTCOME_CV.get(player.position, 0.25)
            + 0.14 * player.injury_risk
            + 0.11 * player.role_risk
            + disagreement,
        )
        with_vorp.append(replace(player, vorp=vorp, outcome_cv=outcome_cv))

    positive_vorp = [max(0.0, player.vorp) for player in with_vorp]
    vorp_scale = max(1.0, max(positive_vorp, default=1.0))
    scored_values = []
    for player in with_vorp:
        projection_signal = float(np.clip(player.vorp / vorp_scale, -0.5, 1.0))
        ecr_signal = math.exp(-(player.ecr - 1.0) / 80.0)
        risk_penalty = 0.10 * player.injury_risk + 0.06 * player.role_risk
        value_score = 100.0 * (0.82 * projection_signal + 0.18 * ecr_signal - risk_penalty)
        scored_values.append(replace(player, value_score=value_score))

    ordered = sorted(scored_values, key=lambda player: player.value_score, reverse=True)
    ranked = tuple(
        replace(player, value_rank=float(rank)) for rank, player in enumerate(ordered, start=1)
    )
    return ValuationResult(
        players=ranked,
        diagnostics=ValuationDiagnostics(
            projection_counts=dict(source_counts),
            adp_counts=dict(adp_counts),
            replacement_points={key: round(value, 2) for key, value in replacements.items()},
        ),
    )
