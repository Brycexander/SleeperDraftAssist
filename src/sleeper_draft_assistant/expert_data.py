"""Validated expert rest-of-season ranking snapshots and consensus values.

The transaction engine treats ranks as an ordering signal. They are deliberately
kept separate from weekly fantasy-point projections.
"""
from __future__ import annotations

from datetime import datetime, timezone
import math
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator, model_validator


Scoring = Literal["STD", "HALF", "PPR"]
POSITIONS = frozenset({"QB", "RB", "WR", "TE"})


class AccuracyEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    season: StrictInt = Field(ge=2020, le=2100)
    horizon: Literal["ros"] = "ros"
    scope: Literal["overall"] = "overall"
    scoring: Scoring
    place: StrictInt = Field(ge=1)
    field_size: StrictInt | None = Field(default=None, ge=2)
    graded_weeks: StrictInt = Field(default=0, ge=0, le=18)
    url: StrictStr

    @field_validator("url")
    @classmethod
    def public_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("accuracy evidence URL must use HTTPS")
        return value

    @model_validator(mode="after")
    def place_is_in_field(self) -> "AccuracyEvidence":
        if self.field_size is not None and self.place > self.field_size:
            raise ValueError("accuracy place cannot exceed field size")
        return self


class ExpertRow(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    player_id: StrictStr = Field(min_length=1, max_length=64)
    position: Literal["QB", "RB", "WR", "TE"]
    overall_rank: StrictInt = Field(ge=1, le=1000)


class ExpertSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expert_id: StrictStr = Field(min_length=1, max_length=128)
    name: StrictStr = Field(min_length=1, max_length=120)
    publisher: StrictStr = Field(min_length=1, max_length=120)
    published_at: datetime
    source_url: StrictStr
    accuracy: AccuracyEvidence | None = None
    accuracy_history: list[AccuracyEvidence] = Field(default_factory=list, max_length=4)
    rows: list[ExpertRow] = Field(min_length=1, max_length=1000)

    @field_validator("published_at")
    @classmethod
    def aware_date(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("published_at must include a timezone")
        return value

    @field_validator("source_url")
    @classmethod
    def source_https(cls, value: str) -> str:
        if not value.startswith("https://"):
            raise ValueError("source URL must use HTTPS")
        return value

    @model_validator(mode="after")
    def unique_rows(self) -> "ExpertSubmission":
        ids = [row.player_id for row in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("an expert submission cannot rank a player twice")
        evidence = self.accuracy_records
        if not evidence:
            raise ValueError("an expert submission requires ROS accuracy evidence")
        seasons = [item.season for item in evidence]
        if len(seasons) != len(set(seasons)):
            raise ValueError("an expert submission cannot repeat an accuracy season")
        return self

    @property
    def accuracy_records(self) -> tuple[AccuracyEvidence, ...]:
        return tuple(self.accuracy_history or ([self.accuracy] if self.accuracy is not None else []))


class ExpertSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    schema_version: Literal[1, 2] = 2
    season: StrictInt = Field(ge=2020, le=2100)
    week: StrictInt = Field(ge=1, le=18)
    horizon: Literal["ros"] = "ros"
    scoring: Scoring
    roster_format: Literal["redraft_1qb"] = "redraft_1qb"
    rank_scope: Literal["overall_qb_rb_wr_te"] = "overall_qb_rb_wr_te"
    fetched_at: datetime
    submissions: list[ExpertSubmission] = Field(min_length=1, max_length=50)

    @field_validator("fetched_at")
    @classmethod
    def fetched_date_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("fetched_at must include a timezone")
        return value

    @model_validator(mode="after")
    def unique_experts(self) -> "ExpertSnapshot":
        ids = [item.expert_id for item in self.submissions]
        if len(ids) != len(set(ids)):
            raise ValueError("an expert snapshot cannot contain duplicate experts")
        return self


@dataclass(frozen=True)
class ExpertExclusion:
    expert_id: str
    reason: str
    detail: str


@dataclass(frozen=True)
class ExpertSelection:
    selected: tuple[ExpertSubmission, ...]
    scores: dict[str, float]
    excluded: tuple[ExpertExclusion, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExpertValue:
    player_id: str
    position: str
    median_rank: float
    best_rank: float
    worst_rank: float
    contributors: tuple[str, ...]
    positional_rank: float
    rank_credit: float
    rank_spread: float = 0.0
    agreement: Literal["high", "medium", "low"] = "high"


@dataclass(frozen=True)
class ExpertBoard:
    values: dict[str, ExpertValue]
    panel_ids: tuple[str, ...]
    universe_size: int
    warnings: tuple[str, ...] = ()
    panel_scores: tuple[tuple[str, float], ...] = ()
    selection_notes: tuple[str, ...] = ()


def rank_credit(rank: float, universe_size: int) -> float:
    if not math.isfinite(rank) or rank <= 0 or universe_size < 1:
        raise ValueError("rank and universe size must be finite and positive")
    return math.log((universe_size + 1) / rank)


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    return float(ordered[midpoint]) if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _accuracy_percentile(evidence: AccuracyEvidence) -> float:
    if evidence.field_size is None:
        raise ValueError("accuracy evidence requires field_size")
    return 1.0 - ((evidence.place - 1) / (evidence.field_size - 1))


def _historical_score(records: list[AccuracyEvidence], season: int) -> float:
    weights = {season - 1: .5, season - 2: .3, season - 3: .2}
    weighted = [(weights[item.season], _accuracy_percentile(item)) for item in records if item.season in weights]
    weight_total = sum(weight for weight, _ in weighted)
    raw = sum(weight * score for weight, score in weighted) / weight_total
    weeks = sum(item.graded_weeks for item in records if item.season in weights)
    years = len(weighted)
    reliability = min(1.0, weeks / 42.0) * (1.0 if years >= 3 else .85)
    return .5 + reliability * (raw - .5)


def select_expert_panel(
    submissions: list[ExpertSubmission], *, season: int, week: int, scoring: str,
    now: datetime, minimum: int = 5, maximum: int = 8,
    max_per_publisher: int = 2, max_age_days: int = 7,
) -> ExpertSelection:
    """Select a current, diverse panel using comparable rolling ROS accuracy."""
    if not 1 <= minimum <= maximum:
        raise ValueError("expert panel minimum must be positive and no greater than maximum")
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    qualified: list[tuple[float, ExpertSubmission]] = []
    excluded: list[ExpertExclusion] = []
    scores: dict[str, float] = {}
    target_seasons = {season - 1, season - 2, season - 3}
    for submission in submissions:
        age = (now - submission.published_at).total_seconds()
        reason = detail = ""
        if age < -300:
            reason, detail = "future_publication", "ranking publication date is in the future"
        elif age > max_age_days * 86400:
            reason, detail = "stale_ranking", f"ranking is older than {max_age_days} days"
        elif len(submission.rows) < 100 or {row.position for row in submission.rows} != POSITIONS:
            reason, detail = "insufficient_coverage", "ranking needs at least 100 rows across QB/RB/WR/TE"
        else:
            records = list(submission.accuracy_records)
            comparable = [item for item in records if item.scoring == scoring and item.season in target_seasons]
            complete_seasons = {item.season for item in comparable}
            if len(complete_seasons) < 2:
                reason, detail = "insufficient_history", "at least two of the last three completed ROS accuracy seasons are required"
            else:
                complete = [item for item in comparable if item.field_size is not None and item.graded_weeks >= 6]
                if len(complete) < 2:
                    if len([item for item in comparable if item.field_size is not None]) < 2:
                        reason, detail = "missing_field_size", "at least two accuracy seasons need a documented field size"
                    else:
                        reason, detail = "insufficient_graded_weeks", "at least two accuracy seasons need six graded ROS weeks"
                    excluded.append(ExpertExclusion(submission.expert_id, reason, detail))
                    continue
                score = _historical_score(complete, season)
                current = next((item for item in records if item.season == season and item.scoring == scoring), None)
                if week >= 7 and current is not None and current.field_size is not None and current.graded_weeks >= 6:
                    completed_weeks = min(current.graded_weeks, week - 1)
                    current_weight = min(.2, completed_weeks / 12 * .2)
                    score = score * (1 - current_weight) + _accuracy_percentile(current) * current_weight
                scores[submission.expert_id] = score
                qualified.append((score, submission))
        if reason:
            excluded.append(ExpertExclusion(submission.expert_id, reason, detail))

    selected: list[ExpertSubmission] = []
    publisher_counts: dict[str, int] = {}
    for _, submission in sorted(qualified, key=lambda item: (-item[0], item[1].expert_id)):
        publisher = submission.publisher.strip().casefold()
        if publisher_counts.get(publisher, 0) >= max_per_publisher:
            excluded.append(ExpertExclusion(submission.expert_id, "publisher_limit", f"already selected {max_per_publisher} experts from {submission.publisher}"))
            continue
        if len(selected) >= maximum:
            excluded.append(ExpertExclusion(submission.expert_id, "panel_limit", f"panel is limited to {maximum} experts"))
            continue
        selected.append(submission)
        publisher_counts[publisher] = publisher_counts.get(publisher, 0) + 1
    warnings: list[str] = []
    if len(selected) < minimum:
        warnings.append(f"Only {len(selected)} qualified experts were available; at least {minimum} are required.")
        selected = []
    return ExpertSelection(tuple(selected), scores, tuple(excluded), tuple(warnings))


def _agreement(median: float, spread: float) -> Literal["high", "medium", "low"]:
    if spread >= max(20.0, median * .4):
        return "low"
    if spread >= max(10.0, median * .2):
        return "medium"
    return "high"


def build_consensus(submissions: list[ExpertSubmission]) -> ExpertBoard:
    if not 3 <= len(submissions) <= 10:
        raise ValueError("a selected expert panel must contain between 3 and 10 experts")
    panel = tuple(item.expert_id for item in submissions)
    if len(set(panel)) != len(panel):
        raise ValueError("duplicate expert IDs")
    minimum = max(3, math.ceil(len(submissions) / 2))
    by_player: dict[str, list[ExpertRow]] = {}
    warnings: list[str] = []
    for submission in submissions:
        for row in submission.rows:
            by_player.setdefault(row.player_id, []).append(row)
    values: dict[str, ExpertValue] = {}
    max_rank = 1
    for rows in by_player.values():
        max_rank = max(max_rank, max(row.overall_rank for row in rows))
    usable: list[tuple[str, list[ExpertRow]]] = []
    for player_id, rows in by_player.items():
        if len(rows) < minimum:
            warnings.append(f"{player_id}: only {len(rows)} expert ranks; need {minimum}")
            continue
        positions = {row.position for row in rows}
        if len(positions) != 1:
            warnings.append(f"{player_id}: conflicting expert positions")
            continue
        usable.append((player_id, rows))
    ordered = sorted(usable, key=lambda pair: (_median([r.overall_rank for r in pair[1]]), pair[0]))
    per_position: dict[str, list[tuple[str, float]]] = {}
    for player_id, rows in ordered:
        median = _median([row.overall_rank for row in rows])
        per_position.setdefault(rows[0].position, []).append((player_id, median))
    positional: dict[str, float] = {}
    for position, group in per_position.items():
        for index, (player_id, _) in enumerate(group):
            same = [rank for _, rank in group]
            rank = same[index]
            ties = [i + 1 for i, candidate in enumerate(same) if candidate == rank]
            positional[player_id] = sum(ties) / len(ties)
    for player_id, rows in ordered:
        median = _median([row.overall_rank for row in rows])
        spread = float(max(row.overall_rank for row in rows) - min(row.overall_rank for row in rows))
        values[player_id] = ExpertValue(
            player_id=player_id, position=rows[0].position, median_rank=median,
            best_rank=float(min(row.overall_rank for row in rows)),
            worst_rank=float(max(row.overall_rank for row in rows)),
            contributors=tuple(sorted({submission.expert_id for submission in submissions if any(r.player_id == player_id for r in submission.rows)})),
            positional_rank=positional[player_id], rank_credit=rank_credit(median, max_rank),
            rank_spread=spread, agreement=_agreement(median, spread),
        )
    return ExpertBoard(values, panel, max_rank, tuple(warnings))


def validate_snapshot_context(snapshot: ExpertSnapshot, *, season: int, week: int, scoring: str,
                              now: datetime, max_age_days: int = 7) -> list[str]:
    if snapshot.season != season or snapshot.week != week or snapshot.scoring != scoring:
        return ["expert snapshot season, week, or scoring does not match the request"]
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    warnings: list[str] = []
    for submission in snapshot.submissions:
        age = (now - submission.published_at).total_seconds()
        if age < -300:
            warnings.append(f"{submission.name}: publication date is in the future")
        elif age > max_age_days * 86400:
            warnings.append(f"{submission.name}: ranking is older than {max_age_days} days")
    return warnings
