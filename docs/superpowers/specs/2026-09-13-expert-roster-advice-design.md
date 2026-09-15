# Expert-based roster advice design

Status: implementation proposal requested by the user; application code has not been changed by this document.

## Intended result

Use a transparent, small consensus of qualified experts' rest-of-season (ROS) rankings to evaluate long-term waiver additions and trades. Use Fantasy Football Tiers for weekly lineup decisions. Keep actual Sleeper game statistics for recent production and workload evidence, but remove Sleeper projected points and season projections from expert-mode transaction decisions.

The user has no FantasyPros subscription. A free, verified data source is therefore a release requirement. An import interface can make development and manual use possible, but is not equivalent to a working automatic source.

## Global constraints

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

## What exists today

Repository: `/home/qwerty/SleeperDraftAssist`.

`weekly.py` loads Sleeper weekly and season stat projections and Fantasy Football Tiers. `WeeklyPlayer.roster_value` is season projected points / 17. `strategy.py` evaluates weekly points plus bench credit plus a small season-strength term. Its opportunity mode blends season/weekly projections, then models recency bias. `management.py` orchestrates these paths. The inline HTML/JavaScript in `team_ui.py` exposes lineup, waiver, and trade advice; `web.py` and `cli.py` expose their inputs.

This is not currently an expert ROS valuation system. Replacing only the label or `roster_value` field would leave projections driving decisions and would fail this specification.

Existing uncommitted implementation files belong to the user. Do not reset, discard, or automatically commit them along with this work.

## Decisions for version 1

### 1. Select experts using evidence

Select five to eight experts from the imported candidate pool. Each candidate must have comparable **overall ROS accuracy** evidence for at least two of the last three completed seasons, with at least six graded weeks per included season. Draft accuracy and weekly-start accuracy do not qualify. Normalize each finish to its documented field size, weight completed seasons 50/30/20 from newest to oldest, and shrink the score toward the field average based on the number of seasons and graded weeks. This avoids treating a short record as equally reliable to three complete seasons.

Beginning in week 7, comparable current-season ROS accuracy may contribute between 10% and 20%, based on six to twelve graded weeks. It never contributes more than 20%. Choose the highest adjusted scores deterministically, cap each publisher at two experts, and require five selected experts. Use a maximum of eight to keep the consensus transparent.

Expert selection scores choose the panel only. Within the chosen panel, each current player rank has equal influence and the consensus is the median. Show the best-to-worst rank spread and a high/medium/low agreement label so users can see uncertainty. The accuracy methodology/scoring and ranking scoring are distinct metadata and both must match the supported league scoring format.

If verifiable multi-year evidence cannot be supplied, withhold expert-mode advice and report each exclusion reason. Do not silently substitute a curated panel.

### 2. Source feasibility is a gate

Public individual-ranking pages exist, but a production-quality, freely accessible, automatically filtered top-10 ROS feed has **not** been verified. A public table, visible download button, or consensus endpoint does not prove complete individual data access. Do not assume the draft consensus loader can recover individual experts from an aggregate.

The first implementation task records reachable URLs, actual season/week, scoring, ranking scope, expert identities, independent accuracy evidence, completeness, and publication timestamps. If no eligible panel is accessible, implement and test the local import path with clearly synthetic fixtures, leave automatic expert advice unavailable, and explain the missing source. Do not declare the user-facing feature fully ready.

### 3. Separate rankings from points

Expert values live in new types and functions. Do not stuff them into `WeeklyPlayer.points`, `roster_value`, or old `weekly_gain` fields. Weekly lineup code continues using its existing contract. A proposed rank-based roster score is an explicit heuristic, not an optimized expectation of future points.

Each expert must provide a comparable **overall QB/RB/WR/TE redraft ROS list** in the same scoring/roster format. A FLEX-only list cannot supply QB value. Positional ranks cannot be summed across positions or treated as overall ranks. Unsupported IDP, dynasty, keeper, superflex, tight-end-premium, or material custom scoring requires an explicit unsupported-format result in v1, not a misleading approximation. Fixed K/DST slots may remain in a league but are not evaluated for ROS transactions; show that scope. Dedicated streamers are outside v1.

Version 1 supports standard, half-PPR, and PPR, one-QB redraft leagues with the conventional scoring profile defined in the implementation plan. Missing expert data is a reason to withhold a comparison, not a reason to drop a player.

### 4. Evaluate actual roster fit

Build a shared consensus once per request. Convert consensus ranks to a monotonic logarithmic **rank credit**, then use the existing exact assignment solver to find the best legal long-term starting combination. Evaluate first backups, second backups, and remaining depth separately. Compare those components lexicographically, starting with starter strength; do not add arbitrary percentages of rankings, points, and workload together.

Use the best unrostered ranked player at each position as a displayed replacement benchmark. Freeze the benchmark and ranking universe before comparing moves. Evaluate each pickup with its required drop and each trade on both rosters. Current injury/bye/game-lock flags do not erase long-term player value. However, transactions must respect the application's supported ownership and transaction-lock policy, with explicit exclusions and no suggestion to override Sleeper restrictions.

### 5. Recent form affects explanations and speculative opportunity labels

Compare positional finishes from completed appearances with the player's positional place in the current expert ROS consensus. This expresses a difference between recent results and current expert expectations; it is **not** a measurement of how accurate an expert's pregame prediction was.

Keep workload change and snap-share change as evidence. Do not automatically apply a second workload bonus to ROS rankings; experts may already have accounted for it. A potential breakout and a role decline must be distinguished from sell-high and buy-low candidates.

One appearance produces `very_low` confidence and no claim about workload direction. Up to four prior weeks are used. Bye weeks, missed appearances, and failed feeds are not zeros. Real zero-point appearances remain valid.

### 6. Trade results distinguish benefit from negotiation

Balanced recommendations must improve both teams under the same expert-rank roster comparison. Opportunity ideas may improve the user's roster while reducing the other team's expert-based value. Such ideas belong in a separate `discussion_candidates` list and require recent-form evidence plus a plausible benefit in the other team's positional needs or recent production.

Do not fabricate market prices or acceptance probabilities. Do not treat “none passed our strict filter” as proof that no good trade exists. Return funnel diagnostics and the best rejected alternatives with their failed conditions; clearly separate these from recommendations.

Version 1 searches all eligible one-for-one trades and a deterministic bounded set of two-for-two packages. Two-for-one deals and picks need additional roster/cut accounting and are explicitly deferred.

### 7. Product behavior and rollout

Add `valuation_source=experts|sleeper` for transactions. Initially preserve the existing source default while expert mode is being developed. Once actual free data and end-to-end checks pass, make experts the default in web and CLI. Keep the existing mode explicitly labeled as the legacy Sleeper model. It must never be an automatic expert-mode fallback.

Hide lineup preference controls on trades/waivers. Expert-mode transactions ignore tiers even if a legacy client submits `mode=tiers`. A lineup panel, if shown beside transactions, is labeled separately and cannot influence trade/waiver ordering.

Show source status, qualified expert count, selection scores, contributors, ranking dates, accuracy qualification, supported scoring, coverage, rank disagreement, rank-credit gains, and concrete rejection reasons. Provide a simple authenticated manual JSON-file import as the free fallback; it is a data refresh workflow, not a hidden production dependency. Persistent cloud storage for those imports is a separate deployment decision; Cloud Run local disk is not durable.

## Validation and limits

Test projection independence by changing all Sleeper forecasts dramatically while holding expert data fixed: expert-mode candidate ordering and gains must remain unchanged. Likewise, changing tiers must not affect transaction results.

Test roster legality, missing data, rank scope/scoring, timestamps, panel selection, one-game form, counterpart benefits, locks, injured-star preservation, diagnostics, and API authentication.

No claim of optimality or better historical performance is authorized without chronological out-of-sample evaluation. Begin collecting dated expert snapshots prospectively. Do not run today's rankings against past games and call that a backtest. A later research phase can compare broad consensus, selected experts, and additional weighting using real historical snapshots.

## Reference evidence checked during planning

- [FantasyPros ROS accuracy methodology](https://www.fantasypros.com/about/faq/football-rest-of-season-accuracy-methodology/) explains the separate ROS competition and its scoring horizon.
- [Public individual ROS rankings example](https://www.fantasypros.com/nfl/fantasy-football-rankings/ros-rb.php) shows expert columns and update dates. It is an example, not proof of a free eligible top-10 overall panel.
- [FantasyPros API access](https://www.fantasypros.com/api-data/) distinguishes sample/development access from production access.
- [FantasyPros AI tools](https://support.fantasypros.com/hc/en-us/articles/55238312588571-What-tools-are-available-in-the-FantasyPros-MCP-Server) documents free-account consensus access; expert filtering and suitability for website automation remain unverified.

These sources can change. Recheck actual availability at execution time and never invent an endpoint or rely on search snippets as a data fixture.
