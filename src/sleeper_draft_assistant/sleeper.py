from __future__ import annotations

from collections import defaultdict
from typing import Any

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
        self, league: dict[str, Any], max_seasons: int = 3
    ) -> dict[str, dict[str, float]]:
        records: list[tuple[str, str, int]] = []
        previous_id = league.get("previous_league_id")
        seasons = 0

        while previous_id and seasons < max_seasons:
            previous = self.league(str(previous_id))
            for draft in self.drafts(str(previous_id)):
                if draft.get("status") != "complete":
                    continue
                for pick in self.picks(str(draft["draft_id"])):
                    user_id = str(pick.get("picked_by") or "")
                    position = normalize_position((pick.get("metadata") or {}).get("position"))
                    if user_id and position:
                        records.append((user_id, position, int(pick["round"])))
            previous_id = previous.get("previous_league_id")
            seasons += 1

        if not records:
            return {}

        league_rounds: dict[str, list[int]] = defaultdict(list)
        manager_rounds: dict[tuple[str, str], list[int]] = defaultdict(list)
        for user_id, position, round_number in records:
            league_rounds[position].append(round_number)
            manager_rounds[(user_id, position)].append(round_number)

        biases: dict[str, dict[str, float]] = defaultdict(dict)
        for (user_id, position), rounds in manager_rounds.items():
            league_mean = sum(league_rounds[position]) / len(league_rounds[position])
            manager_mean = sum(rounds) / len(rounds)
            # Negative values make this manager more likely to select the position.
            biases[user_id][position] = max(-8.0, min(8.0, (manager_mean - league_mean) * 2.0))
        return dict(biases)
