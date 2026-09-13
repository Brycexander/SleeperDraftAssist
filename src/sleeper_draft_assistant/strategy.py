"""Pure, league-slot-aware weekly lineup and roster decision models.

Lineups use exact maximum-weight assignment. Transaction searches are bounded
shortlists, not forecasts of a manager's willingness to trade or claim success.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Any

from .models import normalize_position


@dataclass(frozen=True, slots=True)
class WeeklyPlayer:
    player_id: str
    name: str
    position: str
    team: str
    points: float | None
    roster_value: float = 0.0
    tier: int | None = None
    rank: float | None = None
    upside: float | None = None
    status: str = ""
    eligible_positions: tuple[str, ...] = ()
    locked: bool = False
    source: str = ""
    tier_source: str = ""
    flex_tier: int | None = None
    flex_rank: float | None = None
    bye: bool = False
    rosterable: bool = True


_FLEX = {
    "FLEX": {"RB", "WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
    "REC_FLEX": {"WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "IDP_FLEX": {"DL", "LB", "DB", "DE", "DT", "CB", "S"},
    "DL": {"DL", "DE", "DT"},
    "DB": {"DB", "CB", "S"},
}
_NONSTARTER = {"BN", "IR", "TAXI", "RESERVE"}
_UNAVAILABLE = {
    "OUT", "O", "IR", "PUP", "SUSP", "SUSPENDED", "INACTIVE", "NA",
    "DID_NOT_PLAY", "BYE", "PHYSICALLY_UNABLE_TO_PERFORM", "INJURED_RESERVE",
    "NON_FOOTBALL_INJURY", "NFI",
}


def _finite(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def _eligible(player: WeeklyPlayer, slot: str) -> bool:
    positions = {normalize_position(player.position)} | {
        normalize_position(position) for position in player.eligible_positions
    }
    return bool(positions & _FLEX.get(slot, {slot}))


def _available(player: WeeklyPlayer) -> bool:
    return player.rosterable and not player.bye and player.status.upper().replace(" ", "_") not in _UNAVAILABLE


def _row(player: WeeklyPlayer) -> dict[str, Any]:
    return {
        "player_id": player.player_id, "name": player.name,
        "position": player.position, "team": player.team,
        "points": _finite(player.points), "roster_value": _finite(player.roster_value) or 0.0,
        "tier": player.tier, "rank": _finite(player.rank),
        "flex_tier": player.flex_tier, "flex_rank": _finite(player.flex_rank),
        "status": player.status, "locked": player.locked, "bye": player.bye,
        "source": player.source, "tier_source": player.tier_source,
    }


def _assignment(scores: list[list[float]]) -> list[int]:
    """Rectangular Hungarian assignment, O(rows**2 * columns).

    Callers append one zero-value empty choice per row, so columns >= rows.
    """
    n = len(scores)
    if not n:
        return []
    m = len(scores[0])
    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    assigned, previous = [0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        assigned[0] = row
        column = 0
        distance, used = [math.inf] * (m + 1), [False] * (m + 1)
        while True:
            used[column] = True
            active_row = assigned[column]
            delta, next_column = math.inf, 0
            for candidate in range(1, m + 1):
                if used[candidate]:
                    continue
                cost = -scores[active_row - 1][candidate - 1] - u[active_row] - v[candidate]
                if cost < distance[candidate]:
                    distance[candidate], previous[candidate] = cost, column
                if distance[candidate] < delta:
                    delta, next_column = distance[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[assigned[candidate]] += delta
                    v[candidate] -= delta
                else:
                    distance[candidate] -= delta
            column = next_column
            if not assigned[column]:
                break
        while column:
            preceding = previous[column]
            assigned[column] = assigned[preceding]
            column = preceding
    result = [-1] * n
    for column in range(1, m + 1):
        if assigned[column]:
            result[assigned[column] - 1] = column - 1
    return result


def _preference_scores(players: list[WeeklyPlayer], mode: str) -> tuple[dict[str, float], dict[str, float]]:
    base = {p.player_id: float(p.points) for p in players if _finite(p.points) is not None}
    if mode == "projection":
        return base, {}

    def calibrate(group: list[WeeklyPlayer], flex: bool = False) -> dict[str, float]:
        # Only reorder ranked players with known projections. Unranked players
        # retain their projection; missing forecasts never become invented points.
        ranked = [p for p in group if p.player_id in base and (
            (p.flex_tier is not None or p.flex_rank is not None) if flex
            else (p.tier is not None or p.rank is not None)
        )]
        def key(p: WeeklyPlayer) -> tuple[float, float, float, str]:
            tier, rank = (p.flex_tier, p.flex_rank) if flex else (p.tier, p.rank)
            return (tier if tier is not None else math.inf,
                    rank if rank is not None else math.inf, -base[p.player_id], p.player_id)
        ranked.sort(key=key)
        return dict(zip((p.player_id for p in ranked), sorted((base[p.player_id] for p in ranked), reverse=True)))

    calibrated = dict(base)
    for position in {normalize_position(p.position) for p in players}:
        calibrated.update(calibrate([p for p in players if normalize_position(p.position) == position]))
    # A FLEX-specific ranking is comparable across RB/WR/TE. It is never
    # applied to quarterbacks in SUPER_FLEX.
    flex_scores = calibrate([p for p in players if normalize_position(p.position) in _FLEX["FLEX"]], True)
    return calibrated, flex_scores


def optimize_lineup(
    players: list[WeeklyPlayer], slots: tuple[str, ...], mode: str = "projection",
    current_starters: list[str] | None = None,
) -> dict[str, Any]:
    """Maximize the selected objective over all legal, unlocked assignments.

    Locked starters remain in their existing slot. A locked player outside the
    current starter list stays on the bench. Missing forecasts leave honest gaps.
    """
    if mode == "tier":
        mode = "tiers"
    if mode not in {"projection", "tiers"}:
        raise ValueError("mode must be 'projection' or 'tiers'")
    warnings: list[str] = []
    unique: dict[str, WeeklyPlayer] = {}
    for player in players:
        if player.player_id in unique:
            warnings.append(f"Duplicate player {player.name} was counted only once.")
        else:
            unique[player.player_id] = player
    players = list(unique.values())
    slots = tuple(normalize_position(s) for s in slots if s.upper() not in _NONSTARTER)
    selected: dict[int, WeeklyPlayer] = {}
    frozen_ids: set[str] = set()
    for index, player_id in enumerate(current_starters or []):
        player = unique.get(str(player_id))
        if index >= len(slots) or player is None or not player.locked:
            continue
        if player.player_id in frozen_ids:
            warnings.append(f"Duplicate locked starter {player.name}; check the current lineup.")
            continue
        selected[index] = player
        frozen_ids.add(player.player_id)
        if not _eligible(player, slots[index]):
            warnings.append(f"Locked starter {player.name} is in an ineligible slot; the platform lineup needs review.")
        if not _available(player):
            warnings.append(f"{player.name} is unavailable but locked in the current lineup.")
    if current_starters is None and any(p.locked for p in players):
        warnings.append("Current starters were not supplied; locked players cannot be moved from the bench.")
    candidates = [p for p in players if not p.locked and _available(p)]
    available_slots = [i for i in range(len(slots)) if i not in selected]
    # Calibrate using all eligible roster players, including locked starters,
    # so game locks do not change the relative tier preference of other players.
    scores, flex_scores = _preference_scores([p for p in players if _available(p) or p.player_id in frozen_ids], mode)
    def score(player: WeeklyPlayer, slot: str) -> float | None:
        if not _eligible(player, slot):
            return None
        if slot in {"FLEX", "REC_FLEX", "WRRB_FLEX"} and player.player_id in flex_scores:
            return flex_scores[player.player_id]
        return scores.get(player.player_id)
    matrix: list[list[float]] = []
    for index in available_slots:
        values = [score(p, slots[index]) for p in candidates]
        matrix.append([value if value is not None else -1e12 for value in values] + [0.0] * len(available_slots))
    chosen = _assignment(matrix)
    for index, column in zip(available_slots, chosen):
        if column < len(candidates) and matrix[available_slots.index(index)][column] > -1e11:
            selected[index] = candidates[column]
    starter_ids = {p.player_id for p in selected.values()}
    missing = sum(_finite(p.points) is None for p in selected.values())
    unknown = sum(_finite(p.points) is None and _available(p) and not p.locked for p in players)
    if unknown:
        warnings.append(f"{unknown} available player(s) have no weekly projection and could not be compared; review them manually.")
    if missing:
        warnings.append(f"{missing} locked starter(s) have no projection; the displayed total is only the known subtotal.")
    unfilled = [slot for i, slot in enumerate(slots) if i not in selected]
    if unfilled:
        warnings.append("Unfilled starter slots: " + ", ".join(unfilled) + ". Check eligibility, injuries, locks, and missing projections.")
    if any(p.status.upper() in {"Q", "QUESTIONABLE", "D", "DOUBTFUL"} for p in selected.values()):
        warnings.append("The lineup includes questionable or doubtful players; confirm active status before kickoff.")
    if mode == "tiers":
        if not any(p.tier is not None or p.rank is not None or p.flex_tier is not None or p.flex_rank is not None for p in candidates):
            warnings.append("No usable weekly tiers were supplied; this lineup uses projections.")
        else:
            warnings.append("Tier preferences reorder projected scores within each position; actual projected points are reported separately. Missing tiers fall back to projections.")
    starters = []
    for index, slot in enumerate(slots):
        player = selected.get(index)
        starters.append({"slot": slot, "slot_index": index, **(_row(player) if player else {
            "player_id": None, "name": "Unfilled", "position": "", "team": "", "points": None,
        })})
    return {
        "starters": starters,
        "bench": [_row(p) for p in players if p.player_id not in starter_ids],
        "projected_points": round(sum(_finite(p.points) or 0.0 for p in selected.values()), 4),
        "objective_score": round(sum(score(p, slots[i]) or 0.0 for i, p in selected.items()), 4),
        "objective_label": "Expected weekly points" if mode == "projection" else "Tiers preference (projections break cross-position ties)",
        "missing_projection_count": missing + unknown,
        "unfilled_slots": len(unfilled),
        "warnings": warnings,
    }


def _ids(roster: dict[str, Any], key: str = "players") -> set[str]:
    return {str(p) for p in (roster.get(key) or []) if p and str(p) != "0"}


def _active_ids(roster: dict[str, Any]) -> set[str]:
    return _ids(roster) - _ids(roster, "reserve") - _ids(roster, "taxi")


def _find_roster(rosters: list[dict[str, Any]], roster_id: int) -> dict[str, Any]:
    for roster in rosters:
        if str(roster.get("roster_id")) == str(roster_id):
            return roster
    raise ValueError(f"Roster {roster_id} was not found")


@dataclass(frozen=True)
class _Value:
    weekly: float
    depth: float
    ros: float
    missing: int
    empty: int

    @property
    def total(self) -> float:
        return self.weekly + self.depth + 0.02 * self.ros


def _evaluator(players: dict[str, WeeklyPlayer], slots: tuple[str, ...], mode: str, roster: dict[str, Any]):
    cache: dict[frozenset[str], _Value] = {}
    def evaluate(ids: set[str]) -> _Value:
        key = frozenset(ids)
        if key in cache:
            return cache[key]
        roster_players = [players[p] for p in sorted(ids) if p in players]
        lineup = optimize_lineup(roster_players, slots, mode, [str(p) for p in (roster.get("starters") or [])])
        bench_ids = {p["player_id"] for p in lineup["bench"]}
        # Modest depth credit, with diminishing returns at each position.
        # Do not count injured/bye players as usable cover this week.
        depth = 0.0
        for position in {p.position for p in roster_players}:
            values = sorted((max(0.0, _finite(p.points) or 0.0) for p in roster_players
                if p.player_id in bench_ids and p.position == position and _available(p)
                and any(_eligible(p, s) for s in slots)), reverse=True)
            depth += sum(value * weight for value, weight in zip(values, (0.12, 0.04)))
        value = _Value(lineup["projected_points"], depth,
            sum(max(0.0, _finite(p.roster_value) or 0.0) for p in roster_players),
            lineup["missing_projection_count"], lineup["unfilled_slots"])
        cache[key] = value
        return value
    return evaluate


def _movable(player: WeeklyPlayer) -> bool:
    return not player.locked and player.rosterable


def _ros_safe(outgoing: list[WeeklyPlayer], incoming: list[WeeklyPlayer]) -> bool:
    """Never sacrifice material known season value for a one-week forecast."""
    if outgoing and any((_finite(p.roster_value) or 0.0) <= 0 for p in outgoing + incoming):
        return False
    given = sum(max(0.0, _finite(p.roster_value) or 0.0) for p in outgoing)
    received = sum(max(0.0, _finite(p.roster_value) or 0.0) for p in incoming)
    if given > 0 and received < given * 0.9:
        return False
    # Missing outlooks must not turn an injury/bye or data outage into a drop.
    if any((not _available(p) or _finite(p.points) is None) and p.roster_value <= 0 for p in outgoing):
        return False
    return True


def _gains(before: _Value, after: _Value, prefix: str = "") -> dict[str, float]:
    return {
        prefix + "weekly_gain": round(after.weekly - before.weekly, 4),
        prefix + "depth_gain": round(after.depth - before.depth, 4),
        prefix + "roster_value_gain": round(after.ros - before.ros, 4),
        prefix + "total_gain": round(after.total - before.total, 4),
    }


def _outlook_warnings(involved: list[WeeklyPlayer]) -> list[str]:
    if any(p.roster_value <= 0 for p in involved):
        return ["Longer-term value is missing for one or more players. This is a weekly projection proxy; review future role, bye weeks, and keeper value before acting."]
    return []


def suggest_waivers(
    players: dict[str, WeeklyPlayer], rosters: list[dict[str, Any]], roster_id: int,
    slots: tuple[str, ...], limit: int = 10, mode: str = "projection",
    capacity: int | None = None,
) -> list[dict[str, Any]]:
    """Evaluate add/drop combinations against a bounded free-agent shortlist."""
    if limit <= 0:
        return []
    roster = _find_roster(rosters, roster_id)
    active = _active_ids(roster)
    occupied = set().union(*(_ids(r, "players") | _ids(r, "reserve") | _ids(r, "taxi") | _ids(r, "starters") for r in rosters))
    capacity = max(len(active), len(slots)) if capacity is None else capacity
    if len(active) > capacity:
        return []  # A single add/drop cannot resolve an already-overfull roster.
    starter_slots = tuple(normalize_position(s) for s in slots if s.upper() not in _NONSTARTER)
    available = [p for p in players.values() if p.player_id not in occupied and _movable(p)
        and _available(p) and _finite(p.points) is not None and any(_eligible(p, s) for s in starter_slots)]
    shortlist: dict[str, WeeklyPlayer] = {}
    for position in {p.position for p in available}:
        group = [p for p in available if p.position == position]
        for p in sorted(group, key=lambda p: (p.points or 0.0, p.roster_value), reverse=True)[:15]:
            shortlist[p.player_id] = p
        for p in sorted(group, key=lambda p: (p.roster_value, p.points or 0.0), reverse=True)[:10]:
            shortlist[p.player_id] = p
    drops: list[WeeklyPlayer | None] = [None] if len(active) < capacity else [
        players[p] for p in sorted(active) if p in players and _movable(players[p])
    ]
    evaluate = _evaluator(players, starter_slots, mode, roster)
    before = evaluate(active)
    results = []
    for addition in shortlist.values():
        best = None
        for drop in drops:
            outgoing = [drop] if drop else []
            if not _ros_safe(outgoing, [addition]):
                continue
            after = evaluate((active - ({drop.player_id} if drop else set())) | {addition.player_id})
            if after.total <= before.total + 0.05 or after.empty > before.empty or after.missing > before.missing:
                continue
            gains = _gains(before, after)
            proposal = {
                "add": _row(addition), "drop": _row(drop) if drop else None, **gains,
                "before_projected_points": before.weekly, "after_projected_points": after.weekly,
                "reason": f"Changes the best legal weekly lineup by {gains['weekly_gain']:+.2f} points and depth credit by {gains['depth_gain']:+.2f}.",
                "warnings": _outlook_warnings(outgoing + [addition]) + ["Availability reflects current rosters; waiver priority, FAAB, and transaction locks still apply."],
                "search_scope": "Top 15 weekly projections and top 10 longer-term values per position; every legal drop checked. Season value losses above 10% are excluded.",
            }
            if best is None or proposal["total_gain"] > best["total_gain"]:
                best = proposal
        if best:
            results.append(best)
    return sorted(results, key=lambda r: (r["total_gain"], r["weekly_gain"], r["add"]["player_id"]), reverse=True)[:limit]


def _trade_value(package: tuple[WeeklyPlayer, ...]) -> float:
    return sum(max(0.0, _finite(p.roster_value) or 0.0) for p in package)


def _packages(pool: list[WeeklyPlayer], starters: set[str]) -> list[tuple[WeeklyPlayer, ...]]:
    # Keep pairs crossing positions and starter/bench roles for surplus/need
    # exchanges. Equal counts avoid hidden third-player drops on either team.
    pairs = list(combinations(pool, 2))
    pairs.sort(key=lambda pair: (
        pair[0].position != pair[1].position,
        (pair[0].player_id in starters) != (pair[1].player_id in starters),
        sum(p.points or 0.0 for p in pair),
    ), reverse=True)
    return pairs[:16]


def suggest_trades(
    players: dict[str, WeeklyPlayer], rosters: list[dict[str, Any]], roster_id: int,
    slots: tuple[str, ...], limit: int = 10, mode: str = "projection",
    capacity: int | None = None,
) -> list[dict[str, Any]]:
    """Search balanced one-for-one and bounded two-for-two mutual upgrades."""
    if limit <= 0:
        return []
    own_roster = _find_roster(rosters, roster_id)
    own_ids = _active_ids(own_roster)
    if capacity is not None and len(own_ids) > capacity:
        return []
    slots = tuple(normalize_position(s) for s in slots if s.upper() not in _NONSTARTER)
    def pool(ids: set[str]) -> list[WeeklyPlayer]:
        return [players[p] for p in sorted(ids) if p in players and _movable(players[p])
            and _available(players[p]) and _finite(players[p].points) is not None
            and any(_eligible(players[p], s) for s in slots)]
    own_pool = pool(own_ids)
    own_eval = _evaluator(players, slots, mode, own_roster)
    own_before = own_eval(own_ids)
    own_pairs = _packages(own_pool, _ids(own_roster, "starters"))
    results: list[dict[str, Any]] = []
    for other_roster in rosters:
        other_id = other_roster.get("roster_id")
        if str(other_id) == str(roster_id):
            continue
        other_ids = _active_ids(other_roster)
        if capacity is not None and len(other_ids) > capacity:
            continue
        other_pool = pool(other_ids)
        other_eval = _evaluator(players, slots, mode, other_roster)
        other_before = other_eval(other_ids)
        other_pairs = _packages(other_pool, _ids(other_roster, "starters"))
        for own_packages, other_packages in (
            ([(p,) for p in own_pool], [(p,) for p in other_pool]),
            (own_pairs, other_pairs),
        ):
            for outgoing in own_packages:
                for incoming in other_packages:
                    if not _ros_safe(list(outgoing), list(incoming)) or not _ros_safe(list(incoming), list(outgoing)):
                        continue
                    # Use one scale for both packages. A mixed forecast/ROS
                    # comparison would make the fairness test meaningless.
                    known_ros = all(p.roster_value > 0 for p in outgoing + incoming)
                    given = _trade_value(outgoing) if known_ros else sum(max(0.0, p.points or 0.0) for p in outgoing)
                    received = _trade_value(incoming) if known_ros else sum(max(0.0, p.points or 0.0) for p in incoming)
                    if min(given, received) <= 0 or min(given, received) / max(given, received) < 0.75:
                        continue
                    outgoing_ids, incoming_ids = {p.player_id for p in outgoing}, {p.player_id for p in incoming}
                    if outgoing_ids & incoming_ids:
                        continue
                    own_after = own_eval((own_ids - outgoing_ids) | incoming_ids)
                    if own_after.total <= own_before.total + 0.10 or own_after.empty > own_before.empty or own_after.missing > own_before.missing:
                        continue
                    other_after = other_eval((other_ids - incoming_ids) | outgoing_ids)
                    if other_after.total <= other_before.total + 0.10 or other_after.empty > other_before.empty or other_after.missing > other_before.missing:
                        continue
                    own_gains, other_gains = _gains(own_before, own_after), _gains(other_before, other_after, "partner_")
                    results.append({
                        "partner_roster_id": other_id,
                        "give": [_row(p) for p in outgoing], "receive": [_row(p) for p in incoming],
                        **own_gains, **other_gains,
                        "value_ratio": round(received / given, 4),
                        "reason": f"Your legal lineup changes by {own_gains['weekly_gain']:+.2f} weekly points; their lineup changes by {other_gains['partner_weekly_gain']:+.2f}. Both teams gain after depth and longer-term value are included.",
                        "warnings": _outlook_warnings(list(outgoing + incoming)) + ["Balanced model value is a starting offer, not a prediction of acceptance. Review injuries, future schedules, and manager preferences."],
                        "search_scope": "All eligible one-for-one exchanges plus 16 two-player packages per team; positive modeled value for both teams and at least 90% season-strength balance required. Players with missing season values are protected.",
                    })
    return sorted(results, key=lambda r: (r["total_gain"], min(r["total_gain"], r["partner_total_gain"])), reverse=True)[:limit]
