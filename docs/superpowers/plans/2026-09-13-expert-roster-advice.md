# Expert-Based Roster Advice Implementation Plan

> Historical execution plan. The implemented expert-selection contract was
> subsequently revised to the rolling multi-year method in the companion design
> specification and `README.md`. Those documents and schema version 2 take
> precedence over the single-season examples below.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Work inline; do not spawn agents unless the user authorizes delegation.

**Goal:** Add transparent, free-source expert ROS waiver/trade advice, retain one-appearance buy-low/sell-high evidence, and reserve Fantasy Football Tiers for weekly lineup decisions.

**Architecture:** Introduce typed expert snapshots, a validated source/import boundary, and a pure rank-based transaction engine alongside the legacy projection engine. Keep weekly lineup fields and numerical fantasy-point projections separate from expert rank credits. Gate activation on real source coverage, then expose provenance and search diagnostics in the existing FastAPI/inline-JavaScript app.

**Tech Stack:** Python >=3.12, dataclasses/Pydantic, existing requests/BeautifulSoup, pytest, FastAPI, existing Hungarian assignment helper, inline HTML/JavaScript. No new dependency is required for the initial implementation.

**Spec:** `docs/superpowers/specs/2026-09-13-expert-roster-advice-design.md` (read before this plan).

## Global Constraints

- Python >=3.12; reuse existing dependencies; add no paid service or subscription.
- Preserve existing working-tree edits and existing draft-engine behavior.
- Keep Fantasy Football Tiers out of expert-mode waiver and trade valuation.
- Keep Sleeper projections out of expert-mode candidate selection, valuation, trade balance, and trend classification.
- Sleeper actual statistics, player identities, ownership, league settings, schedules, and injury designations remain permitted inputs.
- Never label ranks or rank-derived scores as projected fantasy points or acceptance probabilities.
- Never invent expert accuracy, publication dates, individual rankings, or missing player values.
- Never silently substitute broad consensus or Sleeper projections when selected-expert data is unavailable.
- One completed prior-week appearance is sufficient for a provisional recent-form signal; current-week partial results remain excluded.
- All recommendations are read-only; do not place trades, claims, or lineup changes.
- No deployment, subscription purchase, new cloud resource, or secret change is authorized by this planning-only request.

## How to execute with a smaller model

1. Read the spec and the current task, including its contracts. Do not redesign the scoring rules mid-task.
2. Complete one task, run its named tests, then record results in this document's checkboxes. Never mark a live-source task complete using synthetic fixtures.
3. If source access fails, follow the specified unavailable/import path. Do not solve it by including unqualified experts or reusing Sleeper projections.
4. Preserve the working tree. If committing is requested during execution, stage only the task's own changes; inspect the diff first. Do not use `git add .`.
5. Do not deploy merely because the plan includes deployment verification instructions. The current user request is to write a plan.

## Current implementation map

| File | Relevant current behavior | Planned change |
|---|---|---|
| `src/sleeper_draft_assistant/weekly.py` | Loads weekly/season projections, metadata, schedule, tiers | Add selective source loading for expert transactions |
| `src/sleeper_draft_assistant/strategy.py` | `WeeklyPlayer`, `_assignment`, `_eligible`, `_ids`, `_active_ids`, `_find_roster`, legacy advice | Reuse stable helpers, keep legacy engine intact |
| `src/sleeper_draft_assistant/trade_history.py` | Fetches actual stats, immediately invokes projection-based trend analysis | Extract raw completed appearances without removing legacy wrapper |
| `src/sleeper_draft_assistant/management.py` | `TeamService.advise` routes all advice | Add explicit expert branch and injection seams |
| `src/sleeper_draft_assistant/web.py` | `TeamRequest`, `/api/team`, auth | Add source selection and authenticated snapshot import |
| `src/sleeper_draft_assistant/cli.py` | `lineup`, `waivers`, `trades` commands | Add transaction-source and import options |
| `src/sleeper_draft_assistant/team_ui.py` | Inline HTML/JS for all advice | Expert cards, source status, diagnostics, imports |
| `src/sleeper_draft_assistant/rankings.py` | Draft-only aggregate ECR via nflreadpy | Do not repurpose as individual ROS data |

New files, each with one responsibility:

- `expert_data.py`: validated snapshot models, expert selection, consensus.
- `expert_sources.py`: approved source retrieval, normalization, persistent-file import/loading.
- `roster_value.py`: rank credits, future-roster assignment, comparison, replacement display.
- `expert_form.py`: completed-appearance positional finishes and recent-form classification.
- `expert_moves.py`: bounded waiver/trade searches and rejection diagnostics.
- `tests/test_expert_data.py`, `test_expert_sources.py`, `test_roster_value.py`, `test_expert_form.py`, `test_expert_moves.py`.
- `tests/fixtures/experts/`: small synthetic JSON/HTML and, only when permitted, minimal source-shape fixtures without full copied ranking tables.
- `scripts/check_expert_source.py`: operator-run live smoke check; never a network unit test.

Do not modify `valuation.py`, `simulator.py`, native C++, draft board loading, authentication secrets, or cloud resources for this feature.

## Locked interfaces and semantics

### Expert snapshot contract

Define these Pydantic models in `expert_data.py`; use strict validation, `extra='forbid'`, finite numbers, timezone-aware dates, and positive ranks. IDs are strings, never names. Store normalized Sleeper IDs in player rows; source-ID mapping belongs in the adapter.

```python
from datetime import datetime
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

class AccuracyEvidence(BaseModel):
    model_config = ConfigDict(extra='forbid')
    season: int = Field(ge=2020, le=2100)
    horizon: Literal['ros'] = 'ros'
    scope: Literal['overall'] = 'overall'
    scoring: Literal['STD', 'HALF', 'PPR']
    place: int = Field(ge=1)
    url: str  # adapter validates public HTTPS provenance; import never fetches it

class ExpertRow(BaseModel):
    model_config = ConfigDict(extra='forbid')
    player_id: str = Field(min_length=1, max_length=64)
    position: Literal['QB', 'RB', 'WR', 'TE']
    overall_rank: int = Field(ge=1, le=1000)

class ExpertSubmission(BaseModel):
    model_config = ConfigDict(extra='forbid')
    expert_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=120)
    publisher: str = Field(min_length=1, max_length=120)
    published_at: datetime
    source_url: str
    accuracy: AccuracyEvidence
    rows: list[ExpertRow] = Field(min_length=1, max_length=1000)

class ExpertSnapshot(BaseModel):
    model_config = ConfigDict(extra='forbid')
    schema_version: Literal[1] = 1
    season: int = Field(ge=2020, le=2100)
    week: int = Field(ge=1, le=18)
    horizon: Literal['ros'] = 'ros'
    scoring: Literal['STD', 'HALF', 'PPR']
    roster_format: Literal['redraft_1qb'] = 'redraft_1qb'
    rank_scope: Literal['overall_qb_rb_wr_te'] = 'overall_qb_rb_wr_te'
    fetched_at: datetime
    submissions: list[ExpertSubmission] = Field(min_length=1, max_length=10)
```

Do not add a curated-panel bypass to these v1 models without the user's explicit panel choice. Verified top-10 membership uses `accuracy.season == snapshot.season - 1`, `place <= 10`, and matching expert identity in the evidence. Duplicate expert IDs are rejected. Adapter evidence must actually establish those facts. Manual import is user-provided evidence, shown as such, not independently certified.

Within one submission, reject duplicate player IDs and conflicting positions. Equal ranks for different players are allowed. Require all four offensive positions and at least 100 total player rows in a usable production submission; test validation helpers with small rows using a separate pure consensus function. A deep list does not prove completeness, so maintain per-player coverage checks as well.

The model snippets define fields; implement validators as part of Task 2. Explicitly reject booleans/numeric strings as ranks, nonfinite numbers, naive dates, and future dates beyond the permitted skew. Parse browser imports with `ExpertSnapshot.model_validate_json(raw)` so ISO timestamps are handled as JSON timestamps. Never treat a typed annotation alone as the missing validation implementation.

Freshness: exact season and current week, `0 <= now - published_at <= 7 days`; allow clock skew up to five minutes only. `fetched_at` is not a substitute for publication time. No “stale but probably fine” fallback. Deduplicate provider submissions before constructing the snapshot, retaining the latest qualified submission per expert and scope.

```python
from dataclasses import dataclass

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

@dataclass(frozen=True)
class ExpertBoard:
    values: dict[str, ExpertValue]
    panel_ids: tuple[str, ...]
    universe_size: int
    warnings: tuple[str, ...]

@dataclass(frozen=True)
class ExpertLoad:
    status: str  # ready | unavailable | stale | insufficient_experts | unsupported_format
    board: ExpertBoard | None
    sources: tuple[dict, ...]
    warnings: tuple[str, ...]

def build_consensus(submissions: list[ExpertSubmission]) -> ExpertBoard: ...
def load_expert_board(cache_dir, *, season, week, scoring, now,
                      snapshot_path=None, refresh=False) -> ExpertLoad: ...
```

The signatures above use ellipses as interface declarations, not incomplete implementation instructions. Each task below defines the required behavior. `cache_dir` and `snapshot_path` are `pathlib.Path`; scoring is a normalized `STD|HALF|PPR` string.

Consensus rules:

1. Qualified panel size must be 3–10.
2. A player needs `max(3, ceil(panel_size / 2))` contributors. Unranked is unknown, not last place or zero.
3. Use the median of available overall ranks. This is our selected-panel consensus method, not a reproduction of FantasyPros ECR.
4. Resolve player ordering by `(median_rank, player_id)` for deterministic output. Tied median ranks share rank credit.
5. Positional rank is the midrank of the player among supported consensus players of the same primary position, ordered by median overall rank. Position conflicts reject that player with a warning.
6. Let `N = max(1, ceil(max(overall_rank across all selected submissions)))`. Freeze N for the request. Define `rank_credit = log((N + 1) / median_rank)`. It is positive and monotonic; it is not fantasy points, a calibrated price, or linear trade value.
7. Store best/worst raw expert rank and contributor IDs. Disagreement is displayed; do not deduct a fabricated uncertainty penalty.

### Roster scoring contract

```python
@dataclass(frozen=True)
class RosterScore:
    starters: float
    backup1: float
    backup2: float
    surplus: float
    missing_ids: tuple[str, ...]
    empty_slots: int

    def key(self) -> tuple[float, float, float, float]:
        return tuple(round(v, 6) for v in
                     (self.starters, self.backup1, self.backup2, self.surplus))

def compare_rosters(before: RosterScore, after: RosterScore) -> int:
    # -1 worse, 0 equal, 1 better; missing/empty safeguards run before this.
    return (after.key() > before.key()) - (after.key() < before.key())

def evaluate_roster(ids: set[str], players: dict[str, WeeklyPlayer],
                    slots: tuple[str, ...], board: ExpertBoard) -> RosterScore: ...
def replacement_ranks(players, rosters, board) -> dict[str, float | None]: ...
```

Use `strategy._assignment` and `_eligible` directly on a rank-credit matrix; do not call `optimize_lineup` with forged projection fields. Use normalized offensive slots only. Append empty dummy columns, forbid illegal assignments with a large negative score, and ensure each player is used once. Long-term assignment ignores `locked`, `bye`, and current `Out/IR` for known ROS-ranked players: those flags do not mean the player has zero future value. It still requires `rosterable` and position eligibility. Do not use actual current starters as a lock on long-term assignment.

From the remaining ranked bench, choose the best first and second backup for each primary position that can fill a league slot. Sum each backup layer separately; leftover credit goes to `surplus`. Multi-position players count once under primary position after starter assignment. Explain that backup layers are a coarse coverage model, not injury probabilities.

`missing_ids` includes unranked offensive players on the active roster. Unchanged missing players may remain, but no recommendation can add, drop, or trade an unranked player. Reject any move increasing missing IDs or empty offensive starter slots. K/DST/IDP are never scored as zero-priced tradable assets.

Freeze replacement ranks from the union of every roster's `players`, `reserve`, `taxi`, and `starters` before searching. For each supported position, display the best unowned qualified player's consensus rank, or null. This is explanatory context, not an extra score or a changing benchmark during simulations.

### History and form contract

```python
@dataclass(frozen=True)
class AppearanceHistory:
    games: dict[str, list[dict]]
    sources: list[dict]
    warnings: list[str]

def load_completed_appearances(cache_dir, season, week, scoring,
                               players, refresh=False) -> AppearanceHistory: ...
def add_positional_finishes(history: AppearanceHistory,
                            players: dict[str, WeeklyPlayer]) -> AppearanceHistory: ...
def analyze_expert_form(player: WeeklyPlayer, value: ExpertValue | None,
                        games: list[dict]) -> dict: ...
```

Reuse actual league scoring from the existing loader. Load appearances for all metadata-known QB/RB/WR/TE players, not only rostered players; positional finishes need a complete comparison population. Every game record carries `week`, `points`, `opportunities`, `snap_share`, and historical `team`. `add_positional_finishes` adds `position_finish` (midrank on descending actual points) for each week/primary position. Sort ties deterministically without breaking their shared numerical finish. If the upstream population for a week is invalid/incomplete, exclude that week and explain; do not assign artificially high finishes from a roster-only subset.

Concrete population checks in the raw loader: validate every row's season/week/category before building ranks; report unmatched player IDs with positive offensive snaps; require at least one valid offensive appearance for every team whose game is marked completed for that week. If any such team is entirely absent, any offensive player cannot be identified, or a fetch/schema validation fails, withhold that week's expert positional finishes. These checks detect obvious truncation but do not prove perfect provider completeness; describe that limitation. Keep the legacy wrapper's behavior unchanged and apply these additional whole-population checks only when producing expert finishes.

Trend thresholds are conservative v1 heuristics, not accuracy claims:

- Need known ROS value, at least one completed appearance, matching historical/current team, and known positive opportunities in each comparison group.
- `expected = value.positional_rank`; `recent = median(position_finish)`; `threshold = max(3, 0.25 * expected)`.
- Hot when `expected - recent >= threshold`; cold when `recent - expected >= threshold`.
- Require `min(2, appearances)` supporting results among the latest three, each beyond the same threshold relative to expected. A lone spike within four games is insufficient.
- For 2–4 appearances, compare latest `min(2, n-1)` appearances with earlier appearances. Workload is targets for WR/TE, rush attempts + targets for RB, pass attempts + rush attempts for QB. Workload change is latest average / earlier average - 1.
- Role decline: workload decrease >20% or snap-share decrease >0.10. Possible breakout: hot with workload increase >20% or snap-share increase >0.10. Otherwise hot => sell_high, cold => buy_low, else neutral.
- One game: workload/snap change null, provisional true, confidence very_low. Two games: provisional true, confidence low. Three/four: provisional false, confidence low, or moderate for four with complete snap data. These labels describe evidence quantity, not a statistical probability.
- Current Out/IR/PUP/suspension => injury_risk, team change => role_change; neither is automatic buy_low. Bye alone does not invalidate prior evidence.
- No multiplier is applied to expert rank credit from any trend.

Output keys: `signal`, `games`, `weeks`, `recent_position_finish`, `expected_position_rank`, `recent_points`, `workload_change`, `snap_share_change`, `confidence`, `provisional`, `reason`. Use null for unavailable numeric values. Signals additionally include `insufficient_data`.

### Search/output contract

```python
@dataclass(frozen=True)
class MoveSearch:
    suggestions: list[dict]
    discussion_candidates: list[dict]
    near_misses: list[dict]
    diagnostics: dict

def suggest_expert_waivers(players, rosters, roster_id, slots, board,
                           *, capacity, limit=10) -> MoveSearch: ...
def suggest_expert_trades(players, rosters, roster_id, slots, board,
                          *, capacity, approach='balanced', form=None,
                          limit=10) -> MoveSearch: ...
```

Shared eligibility: active roster excluding reserve/taxi; owned IDs deduplicated; rosterable; supported offensive position; board value exists. For executable suggestions retain the current application's `not player.locked` transaction guard, but call it `current_game_lock_policy` and do not claim it is a verified universal Sleeper trading rule. For trades this may exclude useful long-term ideas after games start; report it prominently. Unlocked byes or injured players with valid ROS ranks remain valuable and may be acquired; injury-risk trades cannot be speculative rebound recommendations. Overcapacity, disabled transactions, deadlines, and ownership conflicts produce specific diagnostics.

Waivers: all eligible ranked unowned candidates, every eligible ranked drop when full, and the no-drop choice when a slot is available. Choose the best valid drop for each addition; return only improvements under `compare_rosters`, then sort by component gains in tuple order and player ID. Protect unknown-value players, preserve legality, and evaluate reserve/taxi ownership. A full roster must never receive a no-drop recommendation.

Trades: all eligible one-for-one pairs plus at most 32 two-player packages per team. Rank package candidates deterministically: cross-position pairs first, starter/bench combinations second (using the modeled ROS assignment, not weekly projections), then summed rank credit, then IDs. Equal counts only. Deduplicate package IDs. Cache roster evaluations by frozenset of IDs, separately per team/board.

Balanced: both rosters must improve and neither may increase missing/empty slots. Require ratio `min(give_credit, receive_credit) / max(...) >= 0.75`. This generous guard is a heuristic against gross package imbalance, not a market price. Do not reuse the legacy 90% season-projection guard.

Opportunities: evaluate balanced ideas normally, then collect additional speculative ideas when own roster improves, incoming total rank credit exceeds outgoing, package ratio >=0.75, at least one outgoing sell_high and incoming buy_low exist, and the other team either improves a positional starter assignment or receives a better median recent positional finish at a shared position in the outgoing/incoming packages. In a same-position comparison use only qualifying hot/cold players; do not sum ranks across positions. Do not turn the latter test into “their roster improves.” Show the actual other-team score change, even negative. Exclude role_change, role_decline, injury_risk, and insufficient_data assets from speculative packages. Balanced trades do not require history at all.

Common move fields: `kind` (`expert_waiver|expert_balanced_trade|expert_opportunity`), normal `add/drop` or `give/receive` player identity rows, `partner_roster_id` for trades, `own_before`, `own_after`, `own_gain`, `partner_before/after/gain` for trades, `reason`, `warnings`, `search_scope`. Score and gain objects contain `starters`, `backup1`, `backup2`, `surplus`. Expert player rows include `median_rank`, `positional_rank`, `rank_credit`, `contributors`, `best_rank`, `worst_rank`, `status`, `bye`, and form if loaded. Do not populate legacy point-gain fields with these numbers.

Diagnostics shape:

```json
{
  "candidates_evaluated": 120,
  "eligible_own_players": 8,
  "excluded_players": {"missing_expert_value": 2, "current_game_lock_policy": 3},
  "rejections": {"no_own_improvement": 80, "no_partner_improvement": 20,
                 "package_imbalance": 5, "missing_hot_cold_pair": 10,
                 "roster_legality": 0},
  "search_scope": "All eligible 1-for-1 and at most 32 2-player packages per team"
}
```

Numbers above are schema examples only. Count one terminal rejection reason per evaluated candidate using order: legality, package balance, own improvement, partner improvement for balanced; for speculative use legality, package balance, own improvement, incoming credit improvement, hot/cold evidence, counterpart appeal. Maintain separate `balanced_rejections` and `opportunity_rejections` when both paths are searched, rather than adding the same candidate twice to one funnel. Select up to three legal own-improving near misses that failed counterpart/balance conditions, ordered by own gains; include `failed_checks`. Never place legal/data failures in near misses. No actual counterparty is contacted.

## Task 1: Establish free-source feasibility and a reproducible evidence record

**Files:** Create `docs/expert-source-audit.md`; create `scripts/check_expert_source.py` only after selecting a reachable source; inspect `rankings.py`, `weekly.py`.

**Consumes:** User's no-subscription constraint and reference links in the spec.

**Produces:** Evidence-backed adapter choice or an explicit source blocker; actual source URLs and scope, not a guessed API.

- [ ] Read repository instructions and inspect `git status --short`; preserve existing edits.
- [ ] Inspect public ROS rankings and accuracy standings. Use read-only HTTP, verify response content rather than status alone, and do not bypass logins, paywalls, or access controls.
- [ ] Record this exact table, filling cells with observed facts (write `not available` when a fact is absent):

```markdown
| Source URL | Retrieved UTC | Published UTC | Season/week | Scoring/scope | Expert IDs | Accuracy season/place/evidence | Accessible rows | Decision |
|---|---|---|---|---|---|---|---|---|
```

- [ ] Check three eligible experts, all four offensive positions, overall rank comparability, ID mapping, and freshness. A list of three accessible staff experts does not satisfy top-10 qualification automatically.
- [ ] If qualified data exists, write a source smoke script that calls `load_expert_board` after Task 3 and asserts `status == 'ready'`, panel >=3, correct season/week, and source dates <=7 days. It must print only source status, counts, dates, and format; do not dump full tables or credentials.
- [ ] If no qualified data exists, document why and proceed with Tasks 2–8 using the import route. Keep the live-source release gate unchecked. Ask about a concrete verified curated-panel alternative only if the audit identifies one; do not repeatedly ask the user to purchase a subscription.

This source task has no synthetic “passing” substitute. It is complete as an investigation when the outcome and evidence are recorded; automatic sourcing may remain blocked.

## Task 2: Implement snapshot validation and selected-expert consensus

**Files:** Create `expert_data.py`, `tests/test_expert_data.py`.

**Consumes:** Snapshot and consensus contracts above.

**Produces:** `AccuracyEvidence`, `ExpertRow`, `ExpertSubmission`, `ExpertSnapshot`, `ExpertValue`, `ExpertBoard`, `ExpertLoad`, `build_consensus`; helper `rank_credit(rank: float, universe_size: int) -> float`.

- [ ] Write failing pure tests. The following is a runnable core test once the models are imported:

```python
from datetime import datetime, timezone
import math
import pytest
from sleeper_draft_assistant.expert_data import (
    AccuracyEvidence, ExpertRow, ExpertSubmission, build_consensus, rank_credit,
)

def submission(expert_id, a_rank, b_rank):
    return ExpertSubmission(
        expert_id=expert_id, name=expert_id, publisher='Synthetic fixture',
        published_at=datetime(2026, 9, 15, tzinfo=timezone.utc),
        source_url='https://example.invalid/ranks',
        accuracy=AccuracyEvidence(season=2025, scoring='HALF', place=1,
                                  url='https://example.invalid/accuracy'),
        rows=[ExpertRow(player_id='a', position='WR', overall_rank=a_rank),
              ExpertRow(player_id='b', position='WR', overall_rank=b_rank)])

def test_median_consensus_does_not_let_one_extreme_rank_dominate():
    board = build_consensus([submission('e1', 1, 10),
                             submission('e2', 2, 11),
                             submission('e3', 90, 12)])
    assert board.values['a'].median_rank == 2
    assert board.values['a'].rank_credit > board.values['b'].rank_credit
    assert board.values['a'].worst_rank == 90
    assert len(board.values['a'].contributors) == 3

def test_rank_credit_is_monotonic_and_not_a_point_projection():
    assert rank_credit(10, 100) == pytest.approx(math.log(101 / 10))
    assert rank_credit(1, 100) > rank_credit(10, 100) > rank_credit(100, 100) > 0
```

- [ ] Run `.venv/bin/python -m pytest tests/test_expert_data.py -q`; expect missing module/function failures initially.
- [ ] Implement the models and consensus rules exactly. Keep production validation of >=100 rows/all positions in source loading so pure small-fixture consensus tests stay useful.
- [ ] Add explicit tests for duplicate experts, duplicate players, unknown fields, conflicting positions, rank <=0, naive/future timestamps, wrong horizon/scope/season/week, two experts only, rank-11 expert, and a player with only two contributors. Assert exclusion/rejection reasons, not just empty output.
- [ ] Run the full new test file; inspect serialized output using `json.dumps(..., allow_nan=False)`.

## Task 3: Add free source adapter and manual snapshot import

**Files:** Create `expert_sources.py`, `tests/test_expert_sources.py`, fixtures; update `scripts/check_expert_source.py` from Task 1 if a source was verified.

**Consumes:** `ExpertSnapshot`, `ExpertLoad`, `build_consensus`; actual source shape from Task 1.

**Produces:** `load_expert_board(...)`; `import_expert_snapshot(raw: bytes, target_dir: Path, *, season: int, week: int, scoring: str, now: datetime) -> dict`.

- [ ] Write failing tests for atomic import/load and freshness. Generate 120 synthetic rows across QB/RB/WR/TE and three expert submissions using the models from Task 2; all synthetic URLs must end in `.invalid`. Use `tmp_path`, never production paths.
- [ ] Add tests that a 2025 document requested for 2026, an eight-day-old publication fetched today, wrong PPR, a fake HTML login page, duplicate/ambiguous player mapping, and a two-expert panel produce named failure states.
- [ ] Run `.venv/bin/python -m pytest tests/test_expert_sources.py -q` and observe failure.
- [ ] Implement local imports first: parse at most 2 MiB of UTF-8 JSON; validate entire snapshot before writing; write temp file in `target_dir`, flush, then `os.replace` to `expert-ros-{season}-{week}-{scoring}.json`. Reject user-supplied filesystem paths in web payloads. A malformed import must not overwrite the previous valid snapshot.
- [ ] Implement this load sequence:

```python
# In load_expert_board, with actual helpers defined locally in expert_sources.py:
# 1. Derive fixed snapshot filename from validated season/week/scoring.
# 2. If snapshot_path is supplied by CLI/operator, validate that document.
# 3. Otherwise read the fixed imported document, if present.
# 4. If a verified approved adapter is configured, refresh from it when requested
#    or when no valid import exists. Never fetch import source_url values.
# 5. Apply expert eligibility, full-list coverage, publication freshness,
#    exact format, and player identity validation before build_consensus.
# 6. Return ExpertLoad(status, board or None, sources, warnings).
#    No qualified source => unavailable; panel <3 => insufficient_experts.
```

- [ ] If an adapter is feasible, implement it only for the audited host and actual tested response shape. Use 15-second timeouts, at most one retry for transient network/5xx errors, no retries for 401/403/schema failures, and a 1-hour valid-document cache. Reuse existing cache primitives where compatible; include season/week/scoring/panel in identity. Follow redirects only to approved hosts. Missing publication times cannot be replaced by cache creation times.
- [ ] Map source IDs using available exact crosswalks. Ambiguous name-only matches remain unresolved and reported; never guess suffixes or team changes. Preserve unmatched counts.
- [ ] Re-run source tests. Run live smoke only if Task 1 verified a usable adapter. Imported synthetic data proves import mechanics, not expert accuracy or live coverage.

## Task 4: Build the pure ROS roster evaluator

**Files:** Create `roster_value.py`, `tests/test_roster_value.py`; reuse `strategy.py` helpers without modifying the projection algorithms.

**Consumes:** `WeeklyPlayer` identity/eligibility, `ExpertBoard`, league offensive slots.

**Produces:** `RosterScore`, `evaluate_roster`, `compare_rosters`, `replacement_ranks`.

- [ ] Write a known-optimum FLEX assignment test with RB credit 3, RB credit 2, WR credit 4 and slots RB/FLEX: starter score must be 7, not 5. Include a dual-position player and assert one-use-only.
- [ ] Write the central independence test:

```python
from dataclasses import replace

def test_forecasts_byes_and_game_locks_do_not_change_future_rank_value(board, players):
    from sleeper_draft_assistant.roster_value import evaluate_roster
    ids = set(players)
    before = evaluate_roster(ids, players, ('RB', 'WR', 'FLEX'), board)
    changed = {pid: replace(p, points=9999, roster_value=9999, tier=99,
                            flex_rank=999, locked=True, bye=True)
               for pid, p in players.items()}
    after = evaluate_roster(ids, changed, ('RB', 'WR', 'FLEX'), board)
    assert before == after
```

Define `players` and `board` fixtures in this test file with three eligible ranked offensive players matching IDs; use explicit `ExpertValue` rank credits from the contract. Do not borrow fixture state from the old projection tests.

- [ ] Run `.venv/bin/python -m pytest tests/test_roster_value.py -q`; observe failure.
- [ ] Implement direct assignment and backup layers as specified. `compare_rosters` is the exact function above; do not introduce an overall weighted total.
- [ ] Add tests for ranked Out/IR stars retaining long-term value, unranked offensive IDs remaining unknown, empty slots, tied scores, bench-only upgrades, K/DST exclusion, reserve ownership in replacement lookup, and a frozen replacement benchmark across candidate evaluations.
- [ ] Re-run evaluator tests and `tests/test_strategy.py`; legacy lineup/transaction tests must still pass unchanged.

## Task 5: Separate actual history from forecasts and classify expert-relative form

**Files:** Modify `trade_history.py`; create `expert_form.py`, `tests/test_expert_form.py`; extend `tests/test_trade_history.py`.

**Consumes:** Actual-stat fetching/scoring and schedule checks; `ExpertValue.positional_rank`.

**Produces:** `AppearanceHistory`, `load_completed_appearances`, `add_positional_finishes`, `analyze_expert_form`; preserve existing `load_trade_history` signature and behavior for legacy mode.

- [ ] Add a regression test proving the legacy wrapper returns its previous projection-based trend results after extraction.
- [ ] Write this one-game behavior test:

```python
def test_one_completed_appearance_is_provisional_against_expert_rank():
    from sleeper_draft_assistant.strategy import WeeklyPlayer
    from sleeper_draft_assistant.expert_data import ExpertValue
    from sleeper_draft_assistant.expert_form import analyze_expert_form
    p = WeeklyPlayer('p', 'Fixture WR', 'WR', 'CHI', None)
    value = ExpertValue('p', 'WR', 35, 30, 40, ('e1','e2','e3'), 20, 1.0)
    games = [{'week': 1, 'points': 28, 'opportunities': 8,
              'snap_share': .8, 'team': 'CHI', 'position_finish': 3}]
    form = analyze_expert_form(p, value, games)
    assert form['signal'] == 'sell_high'
    assert form['confidence'] == 'very_low'
    assert form['provisional'] is True
    assert form['workload_change'] is None
```

- [ ] Run the history/form tests and observe missing function failures.
- [ ] Extract the existing actual-history loop with its checks intact. Make legacy `load_trade_history` call the raw loader and then the existing `analyze_trade_trend`. The new expert branch calls raw loader and `analyze_expert_form` only.
- [ ] Add positional midranks using all offensive appearances. For tied scores `[20, 20, 10]`, expected finishes are `[1.5, 1.5, 3]`. Add tests that including an unrostered high scorer changes the rostered player's finish correctly; filtering to rostered players must not happen before ranking.
- [ ] Implement threshold/workload rules from the contract. Keep current expert expectation distinct from pregame prediction accuracy. Do not import or read `roster_value` or projected `points` in `expert_form.py`.
- [ ] Add cases: cold with stable usage => buy_low; cold with falling usage => role_decline; hot with increasing usage => possible_breakout; Out => injury_risk; historical team mismatch => role_change; bye plus valid past appearances => supported; one spike among otherwise ordinary results => neutral.
- [ ] Add no-current-week, no-prior-season, postponed/unverified schedule, real zero points, absent snaps, missing opportunities, duplicate stats, failed feed, and empty history cases. Missing weeks reduce evidence rather than becoming zeros.
- [ ] Run `.venv/bin/python -m pytest tests/test_expert_form.py tests/test_trade_history.py tests/test_trade_trends.py -q`.

## Task 6: Implement ranked transaction search with meaningful empty states

**Files:** Create `expert_moves.py`, `tests/test_expert_moves.py`.

**Consumes:** `evaluate_roster`, `compare_rosters`, rank values, form dictionaries, ownership and capacity.

**Produces:** `MoveSearch`, `suggest_expert_waivers`, `suggest_expert_trades` and the specified result fields.

- [ ] Write failing tests using small explicit rank-credit boards. Required numeric fixtures:

```text
Balanced cross-position trade, slots RB/WR:
Team A: RB a1=5, RB a2=4, WR a3=1. Starter credit=6.
Team B: WR b1=5, WR b2=4, RB b3=1. Starter credit=6.
Exchange a2 for b2 => both starter credits=9; package ratio=1.
Expected: expert_balanced_trade with +3 starters on each team.

Waiver, slots RB/WR, capacity=3:
Own RB a1=5, WR a3=1, RB bench=2. Free WR w=3.
Expected: add w, drop a3, starter gain=2 and backup RB retained.
If a3 has no expert value, never recommend dropping a3 automatically.

Speculative same-position trade, slot WR:
Own hot WR h=3; their cold WR c=3.5; give/receive ratio=3/3.5.
h ROS WR20 with recent finish WR3; c ROS WR10 with recent finish WR30.
Expected: own starter +0.5, partner -0.5; discussion_candidates only.
It must not appear as a balanced recommendation or claim partner improvement.
```

- [ ] Add a forecast-independence test: run both searches, replace every `WeeklyPlayer.points` and `roster_value` with reversed/extreme forecasts and all tiers with different values, rerun, and assert identical results (excluding separately rendered weekly lineup context).
- [ ] Run `.venv/bin/python -m pytest tests/test_expert_moves.py -q`; observe failure.
- [ ] Implement waiver enumeration, cached evaluations, exact drop accounting, and deterministic package selection. Apply the explicit comparison/legality/balance gates. Do not use `_packages` unchanged: it currently sorts using weekly projected points.
- [ ] Implement speculative ideas in their separate list, with no invented perceived-value score. Balanced search must run even when history is absent or no hot/cold pair exists.
- [ ] Implement funnel counts and up to three near misses with the failed conditions. Verify a no-trade result distinguishes no qualified expert data, no own sell-high, no incoming buy-low, game-lock exclusions, no own improvement, and no counterparty improvement.
- [ ] Add tests for reserve/taxi/other ownership, duplicates, overcapacity, missing-value drop protection, bye/injured-star retention, empty-slot protection, locked transaction assets, low-value package stacking, limit=0, deterministic ties, two-for-two size, and no unranked asset in a proposal.
- [ ] Re-run the test file. Measure search time on a synthetic 12-team, 18-player roster fixture and record it; if >5 seconds locally, profile and cache evaluations before increasing caps. Never cut completeness silently to meet a timing target.

## Task 7: Wire service, selective data loading, API and CLI

**Files:** Modify `weekly.py`, `management.py`, `web.py`, `cli.py`, `tests/test_weekly.py`, `tests/test_management.py`, `tests/test_cli.py`.

**Consumes:** `ExpertLoad`, `MoveSearch`, raw history; existing auth and league resolution.

**Produces:** Explicit source-aware transaction responses, supported-format checks, import route/CLI.

New signatures:

```python
# Append keyword-only source controls; existing callers retain old behavior.
def load_weekly_players(cache_dir, season, week, scoring, refresh=False,
                        confirm_tiers_week=False, *, include_projections=True,
                        include_tiers=True): ...

# Append constructor injection seams after existing client/loader/cache_dir args:
# expert_loader=load_expert_board, history_loader=load_completed_appearances
# Append advise keyword argument:
# valuation_source: str = 'sleeper'  # switch default only at Task 9 release gate

def expert_format(league: dict) -> tuple[str | None, str | None]: ...
# returns (STD|HALF|PPR, None) or (None, explicit unsupported reason)
```

`expert_format` belongs in `management.py`. Support exactly one QB starting slot, no SUPER_FLEX, no offensive IDP FLEX, redraft type 0, no keepers. Require reception scoring 0, .5, or 1; pass yards .04, passing TD 4, interceptions -2, rush/receiving yards .1, rush/receiving TD 6, fumbles lost -2, and standard two-point conversions 2. Missing scoring keys mean zero (do not silently assume conventional values). Other nonzero offensive bonus/TE premium fields or changed basic coefficients are unsupported in v1. Ignore defensive/kicking scoring fields for this check. Record which fields caused the rejection. This explicit limitation prevents claiming arbitrary league scoring can be reconstructed from ranks.

The existing test league uses 6-point passing TDs; expert-mode tests must override that to 4 or assert unsupported. Preserve existing legacy tests using 6-point TDs. If actual user leagues require unsupported scoring, report it as a source/format gap rather than activating incorrect rankings.

- [ ] Write service tests that inject an `ExpertLoad` and use a loader spy. For expert trades/waivers assert `include_projections=False` and `include_tiers=False`; lineup/legacy requests retain existing behavior.
- [ ] Write a test whose legacy `suggest_trades`/`suggest_waivers`/`analyze_trade_trend` functions raise if called from expert mode. Missing expert data must return `suggestions=[]`, named `expert_status`, and a warning without touching legacy scorers.
- [ ] Run relevant management/weekly tests; observe failure.
- [ ] Implement selective weekly loading: metadata and schedule remain loaded; skipped forecasts leave `points=None`, `roster_value=0`; skipped tiers perform no tier fetch. Preserve available player status/eligibility and default arguments for existing code.
- [ ] Suppress legacy missing-projection/season-value warnings when those sources were intentionally skipped. They are not outages in expert mode. Retain real identity, injury, schedule, and ownership warnings. Test that a ready expert response does not claim its advice is incomplete merely because `points=None`.
- [ ] Implement the source branch after league validation/ownership and before legacy valuation. Retain disabled-adds, disabled-trades, deadline, capacity, and renewal handling. Record those as transaction-block reasons rather than no-good-trade conclusions.
- [ ] For expert transactions set `lineup=None` and `method='Expert rest-of-season roster comparison'`; the UI should not compute a fake weekly lineup from missing projections. Lineup requests still return the existing lineup object. Add fields `valuation_source`, `expert_status`, `expert_panel`, `expert_coverage`, `discussion_candidates`, `near_misses`, `diagnostics`, `replacement_ranks`; preserve existing `warnings`/`sources` and league identity fields.
- [ ] Add `valuation_source: Literal['experts','sleeper']` to `TeamRequest`, pass to `advise`. Ignore tiers preference for expert transactions with clear response method; do not reject older clients solely for submitting `mode='tiers'`.
- [ ] Add authenticated `POST /api/team/expert-snapshot`. Body is the snapshot JSON, not a URL or path. Enforce 2 MiB while streaming the body, validate before storing under the configured operator directory, and return status/count/date metadata. No external fetch on import; invalid bytes =>400, invalid schema/context=>422, too large=>413, unauthenticated=>401. Import requests use `application/json` and same-origin browser requests; do not enable permissive CORS.
- [ ] Add CLI `--valuation-source experts|sleeper` only to waivers/trades. Add `import-expert-rankings PATH` using `import_expert_snapshot`; derive context from the validated file and check against current NFL state, avoiding draft initialization. Local operator path is allowed; browser path input is not.
- [ ] Use environment `EXPERT_SNAPSHOT_DIR` as a server-owned directory; default to a subdirectory of existing cache. Document local disk durability limits. Do not create a bucket or add a cloud service here.
- [ ] Add auth, malformed input, unsupported-format, disabled-league, missing-source, successful expert advice, and legacy regression API tests using the existing ASGI `_http` helper; test the upload route without live data or secrets.
- [ ] Run `.venv/bin/python -m pytest tests/test_weekly.py tests/test_management.py tests/test_cli.py -q`.

## Task 8: Present expert evidence, source imports, and honest trade outcomes

**Files:** Modify `team_ui.py`, `README.md`; extend web/management HTML response tests.

**Consumes:** Task 7 report contract; existing `esc`, `metric`, rendering helpers.

**Produces:** Source-specific renderers, source panel/import UI, actionable empty states.

- [ ] Add a source selector on transactions only: `Selected experts — rest of season` and `Sleeper projections — legacy`. Keep the existing default until Task 9. Hide the lineup-preference control outside the Lineup tab. Keep Fantasy Football Tiers available within Lineup.
- [ ] Render expert responses before dereferencing `data.lineup`:

```javascript
function render(data) {
  if (data.valuation_source === 'experts' && data.action !== 'lineup') {
    renderExpertReport(data);
    return;
  }
  renderLegacyReport(data);
}
```

Move existing rendering body into `renderLegacyReport` with unchanged behavior. Define `renderExpertReport` in the same script; it uses the specified expert fields, hides the lineup/bench sections, and shows league, selected week, expert count, and source status. All names, reasons, and URLs are escaped or assigned through `textContent`; clickable sources permit HTTPS only.

- [ ] In expert move cards label starter/backups/depth gains as rank credits. Show outgoing/incoming expert ranks and injuries/byes. Do not use “points gained,” “acceptance chance,” or “optimal trade.”
- [ ] Render `discussion_candidates` under `Speculative trade ideas` with the other team's actual rank-credit change and explanation of recent-form appeal. Show one-game signals as `Provisional — one appearance` and workload as `Unknown`, never `0%` for null.
- [ ] Render `near_misses` only in `Why other offers were excluded`, with failed checks visible. Empty example: `No offers passed these filters. 8 eligible players; 3 excluded by the current game-lock policy; 20 offers improved your roster but not the other roster.` Use actual diagnostics; do not invent counts.
- [ ] Source panel shows contributor names, ROS accuracy season/place, accuracy scoring, current ranking scoring, publication dates, panel/coverage counts, and manual-vs-automatic provenance. If fewer than three qualify, state that fact and keep results unavailable.
- [ ] Add local JSON file selection under source options; parse the file client-side only to enforce size and valid JSON, POST to the authenticated import route, show validation errors, then refresh advice. Do not expose server filesystem paths. Include a downloadable synthetic schema example clearly labeled `Example only — not real rankings` if a template is provided.
- [ ] Update README with source acquisition/import steps, supported formats, thresholds/heuristic limitations, strict fallback behavior, current-game-lock exclusions, all-four-position requirement, one-appearance semantics, rank-credit units, and the separation of lineup vs transaction advice.
- [ ] Validate JavaScript syntax with available local tooling (the environment previously used cached QuickJS); execute rendered-page cases for ready expert, missing source, empty trades, speculative ideas, import error, and legacy lineup. A syntax-only check is not a browser interaction test. If no browser runner is available, report that limitation and perform manual local browser checks before release.

## Task 9: Final verification and activation gate

**Files:** All changed files; update source audit and README with actual results, not projections of success.

**Consumes:** Completed implementation, actual source evidence, current user authorization.

**Produces:** Verified local feature and a precise statement of live-source/deployment status.

- [ ] Run `.venv/bin/python -m pytest -q` and `git diff --check`. Use the environment's approval mechanism if multiprocessing/socket tests require escalation. Record actual passing counts; the previous baseline was 117 tests, not the required future total.
- [ ] Recheck these invariants in tests: no Sleeper forecast/tiers influence on expert transactions; legal roster assignments; missing values protected; experts/source dates/scoring validated; balanced requires both teams to gain; speculative offers visibly separate; one completed appearance works; no-data results explain why.
- [ ] Run an actual free-source smoke test and one actual supported league through `TeamService.advise`, without changing Sleeper state. Confirm selected expert identities and dates, player mapping, coverage of every proposed asset, source scoring, and recommendation response. Save counts/status in the audit. Synthetic fixture tests cannot check this box.
- [ ] If source or format coverage is unavailable, keep expert mode selectable but clearly unavailable without an imported valid snapshot, preserve legacy default, and report the exact remaining dependency. Do not claim the user goal is fully achieved.
- [ ] Only when free live data or an explicitly accepted manual import workflow is verified, change web, CLI, and service transaction defaults consistently to experts; update tests that intentionally use legacy mode to request it explicitly. Run the affected tests again.
- [ ] Review final diff; avoid unrelated refactors or changes to the original draft algorithm. Do not commit the user's pre-existing edits accidentally.
- [ ] Hand off with source status, supported/unsupported user leagues, test results, model limitations, and whether anything is deployed. For this planning request, nothing is deployed.

If a later user instruction authorizes deployment, use the existing service only:

```bash
gcloud run deploy sleeper-draftassist --source . --project project-2a7ad8ae-c095-445a-a45 --region us-central1 --quiet
```

First verify project/service, preserve existing secret references and runtime settings, and resolve persistent snapshot storage if using manual imports. After deployment check new revision/traffic, login, authentication, and the new request schema. Do not retrieve secrets or forge a production session for testing. The service URL is `https://sleeper-draftassist-157686765830.us-central1.run.app`. The last confirmed revision before this plan was `sleeper-draftassist-00003-cft`; inspect live state rather than assuming it is still current.

## Deferred evaluation phase: earn stronger weighting claims

This is a separate research task, not a requirement to ship an honestly labeled v1 heuristic.

Persist timestamped expert snapshots prospectively with source, expert IDs, publication/retrieval dates, scoring, league roster snapshot, and algorithm version. Later evaluate only decisions with all inputs available before their cutoff. Compare broad consensus, selected-panel median, and the full roster-aware method on the same supported player pool. Report missing coverage and abstentions as well as ranking error, roster improvement, and replacement-relative results. Train/tune thresholds on earlier seasons and evaluate on untouched later weeks/seasons; never select “top experts” using the season being tested. Trades' acceptance cannot be backtested from ranks without actual offer/acceptance data. Do not fabricate that data.

## Copy-and-paste handoff prompt

> Implement `docs/superpowers/plans/2026-09-13-expert-roster-advice.md` in `/home/qwerty/SleeperDraftAssist`, reading the linked design first. Work task by task inline and preserve all existing edits. The user has no paid fantasy-data subscription. Verify the free selected-expert source first; do not claim automatic top-expert rankings exist without evidence. Keep expert trade/waiver valuation independent of Sleeper forecasts and Fantasy Football Tiers. Keep one-game form provisional, separate speculative trades from balanced recommendations, and explain rejected candidates. Run each task's tests and report any source/format limitation honestly. Do not deploy or change cloud resources unless separately authorized. Stop inventing alternatives when a required source fact is missing: follow the unavailable/import behavior in the plan.
