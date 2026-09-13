from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .models import LeagueContext, LeagueRules, normalize_position


class SleeperClient:
    BASE_URL = "https://api.sleeper.app/v1"

    def __init__(self, timeout: float = 15.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "sleeper-draft-assistant/0.1"
        retries = Retry(
            total=3,
            backoff_factor=0.4,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
        )
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get(self, path: str) -> Any:
        response = self.session.get(f"{self.BASE_URL}{path}", timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def league(self, league_id: str) -> dict[str, Any]:
        return self.get(f"/league/{league_id}")

    def user(self, username: str) -> dict[str, Any]:
        return self.get(f"/user/{username}")

    def draft(self, draft_id: str) -> dict[str, Any]:
        return self.get(f"/draft/{draft_id}")

    def drafts(self, league_id: str) -> list[dict[str, Any]]:
        return self.get(f"/league/{league_id}/drafts")

    def picks(self, draft_id: str) -> list[dict[str, Any]]:
        return self.get(f"/draft/{draft_id}/picks")

    def traded_picks(self, draft_id: str) -> list[dict[str, Any]]:
        return self.get(f"/draft/{draft_id}/traded_picks")

    def users(self, league_id: str) -> list[dict[str, Any]]:
        return self.get(f"/league/{league_id}/users")

    def rosters(self, league_id: str) -> list[dict[str, Any]]:
        return self.get(f"/league/{league_id}/rosters")

    def nfl_state(self) -> dict[str, Any]:
        return self.get("/state/nfl")

    def user_leagues(self, user_id: str, season: int) -> list[dict[str, Any]]:
        return self.get(f"/user/{user_id}/leagues/nfl/{season}")

    def matchups(self, league_id: str, week: int) -> list[dict[str, Any]]:
        return self.get(f"/league/{league_id}/matchups/{week}")

    def sync(self, league_id: str, username: str) -> LeagueContext:
        league = self.league(league_id)
        user = self.user(username)
        draft_id = league.get("draft_id")
        if not draft_id:
            drafts = self.drafts(league_id)
            if not drafts:
                raise ValueError(f"League {league_id} does not have a draft")
            draft_id = drafts[0]["draft_id"]

        draft = self.draft(str(draft_id))
        user_id = str(user["user_id"])
        draft_order = draft.get("draft_order") or {}
        if user_id not in draft_order:
            raise ValueError(f"User {username!r} is not assigned a slot in this draft")

        draft_slot = int(draft_order[user_id])
        slot_to_roster = draft.get("slot_to_roster_id") or {}
        roster_id = int(slot_to_roster.get(str(draft_slot), draft_slot))
        roster_positions = tuple(
            normalize_position(position) for position in league["roster_positions"]
        )
        rules = LeagueRules(
            teams=int(draft["settings"]["teams"]),
            rounds=int(draft["settings"]["rounds"]),
            roster_positions=roster_positions,
            scoring={key: float(value) for key, value in league["scoring_settings"].items()},
        )
        return LeagueContext(
            league_id=league_id,
            username=username,
            user_id=user_id,
            roster_id=roster_id,
            draft_slot=draft_slot,
            league=league,
            draft=draft,
            picks=self.picks(str(draft_id)),
            traded_picks=self.traded_picks(str(draft_id)),
            users=self.users(league_id),
            rosters=self.rosters(league_id),
            rules=rules,
        )

    def manager_position_biases(
        self,
        league: dict[str, Any],
        max_seasons: int = 3,
        additional_league_ids: Iterable[str] = (),
    ) -> dict[str, dict[str, float]]:
        records: list[tuple[str, str, int, float]] = []
        roots: list[tuple[dict[str, Any], float, bool]] = [(league, 1.0, False)]
        current_league_id = str(league.get("league_id") or "")
        for league_id in dict.fromkeys(str(value) for value in additional_league_ids):
            if not league_id or league_id == current_league_id:
                continue
            try:
                roots.append((self.league(league_id), 0.65, True))
            except requests.RequestException:
                continue

        seen_drafts: set[str] = set()
        for root, league_weight, include_root in roots:
            current = root if include_root else None
            previous_id = (
                root.get("league_id") if include_root else root.get("previous_league_id")
            )
            age = 0
            while previous_id and age < max_seasons:
                if current is None or str(current.get("league_id")) != str(previous_id):
                    try:
                        current = self.league(str(previous_id))
                    except requests.RequestException:
                        break
                season_weight = league_weight * 0.70**age
                try:
                    drafts = self.drafts(str(previous_id))
                except requests.RequestException:
                    break
                for draft in drafts:
                    draft_id = str(draft.get("draft_id") or "")
                    if draft.get("status") != "complete" or draft_id in seen_drafts:
                        continue
                    seen_drafts.add(draft_id)
                    try:
                        picks = self.picks(draft_id)
                    except requests.RequestException:
                        continue
                    for pick in picks:
                        user_id = str(pick.get("picked_by") or "")
                        position = normalize_position(
                            (pick.get("metadata") or {}).get("position")
                        )
                        if user_id and position:
                            records.append(
                                (user_id, position, int(pick["round"]), season_weight)
                            )
                previous_id = current.get("previous_league_id")
                current = None
                age += 1

        if not records:
            return {}

        league_rounds: dict[str, list[tuple[int, float]]] = defaultdict(list)
        manager_rounds: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
        for user_id, position, round_number, weight in records:
            league_rounds[position].append((round_number, weight))
            manager_rounds[(user_id, position)].append((round_number, weight))

        biases: dict[str, dict[str, float]] = defaultdict(dict)
        for (user_id, position), rounds in manager_rounds.items():
            league_total_weight = sum(weight for _, weight in league_rounds[position])
            manager_total_weight = sum(weight for _, weight in rounds)
            league_mean = sum(
                round_number * weight
                for round_number, weight in league_rounds[position]
            ) / league_total_weight
            manager_mean = sum(
                round_number * weight for round_number, weight in rounds
            ) / manager_total_weight
            reliability = manager_total_weight / (manager_total_weight + 20.0)
            # Negative values make this manager more likely to select the position.
            raw_bias = (manager_mean - league_mean) * 2.0
            biases[user_id][position] = max(
                -8.0, min(8.0, raw_bias * reliability)
            )
        return dict(biases)
