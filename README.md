# Sleeper Draft Assistant

Monte Carlo draft recommendations tailored to a Sleeper league's scoring,
roster slots, draft order, completed picks, and prior manager tendencies.

The defaults are configured for `brycexander` in The Unemployables
(`1387590026778411008`), with Hooligans (`1389738046894657536`) available as a
saved league and Shield AI After Hours (`1401668384990494720`) available as
`shield-ai`. Player value combines league-scored Sleeper projections, three
seasons of nflverse results, and the current full-PPR Expert Consensus Ranking
board. Sleeper PPR ADP and your leagues' draft histories model availability
separately.

## Setup

```bash
cd /home/brycexander/sleeper-draft-assistant
UV_CACHE_DIR=.uv-cache uv sync --extra dev --python /usr/bin/python3
```

## Commands

Check the synced settings and draft slot:

```bash
uv run sleeper-draft league
```

Inspect the league-adjusted value board and its inputs:

```bash
uv run sleeper-draft values --top 50
uv run sleeper-draft values --position RB --top 25
```

Run a recommendation before or during the draft:

```bash
uv run sleeper-draft recommend
```

The recommendation output includes separate timings for model preparation and
Monte Carlo simulation, plus their combined runtime and selected engine.

Use the native C++ engine for 10,000 runs apiece across 15 candidates:

```bash
uv run sleeper-draft --league hooligans recommend \
  --engine cpp --candidates 15 --runs-per-candidate 10000 --workers auto
```

`--engine auto` is the default and falls back to Python with a warning if the
native extension is unavailable. Use `--engine python` to compare against the
reference implementation. Before your turn, `--runs-per-candidate` multiplies
by `--candidates` to set the total rollout count; on the clock, each candidate
receives the requested number of rollouts.

Run the offline 150,000-rollout performance gate with:

```bash
uv run python benchmarks/benchmark_native.py
```

The native extension requires C++23. On Ubuntu 20.04, install the current Clang
compiler from LLVM's official repository, then rebuild:

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 22
sudo apt-get install -y lld-22 libc++-22-dev libc++abi-22-dev
CC=clang-22 CXX=clang++-22 uv sync --extra dev --reinstall
```

Run the same league-aware advice for Hooligans:

```bash
uv run sleeper-draft --league hooligans recommend
```

Run Shield AI After Hours after its commissioner assigns the draft order:

```bash
uv run sleeper-draft --league shield-ai recommend
```

Summarize the most common simulated teams and estimate playoff probability:

```bash
uv run sleeper-draft analyze --simulations 1000
```

The analysis includes finish probabilities, expected record, common players,
common four-pick openings, final positional builds, and one representative median
draft. It uses the league's playoff-team count and regular-season length.

The default weekly team-score variation is 22%. It can be changed for a more or
less volatile season model:

```bash
uv run sleeper-draft analyze --weekly-variance 0.25
```

Watch the board and rerun automatically whenever a pick is made:

```bash
uv run sleeper-draft watch --simulations 1000 --interval 5
```

Force a fresh ranking download or use more rollouts:

```bash
uv run sleeper-draft recommend --simulations 5000 --refresh-rankings
```

Global settings go before the command when targeting an arbitrary league:

```bash
uv run sleeper-draft --league-id LEAGUE_ID --username USERNAME recommend
```

## Reading The Result

Before your turn, `Available` is the estimated chance a player reaches you and
`AI selects` is how often the adaptive policy chooses him after simulating the
picks ahead. Once you are on the clock, every listed candidate is forced into
the roster and the remaining draft is rolled out to compare the completed teams.

`Top roster` is the rate at which your final roster has the highest sampled
projection score in the simulated league. It is useful for comparing choices
inside this model, but it is not a literal championship probability.

On-clock results also report three explanatory components:

- `Lineup +` is the candidate's projected gain over your current optimal
  starters. Bench credit is excluded from this explanatory metric.
- `Next turn` is the ADP-based chance the player survives to your following
  selection, conditional on being available now.
- `Wait cost` is the probability-weighted VORP drop to the next player in the
  same positional tier.

## Model Inputs

Player quality starts with raw Sleeper season stat projections scored against
the league's actual scoring settings. The same scoring rules are applied to the
previous three regular seasons of nflverse player statistics. Actual season
production is weighted 1.00/0.70/0.49 by recency and blended into the current
projection according to games-played reliability and current role stability;
the maximum historical weight is 25% for RB/WR, 22% for TE, 20% for QB, and 10%
for K. Network failures fall back to the newest local cache or projection-only
values.

The blended projection is converted to value over replacement using the actual
team count, required starters, and flex demand. The displayed score is broken
into an 82-point normalized VORP component, an 18-point ECR prior, and a
separate injury/role risk deduction. The Monte Carlo recommendation is still
ranked by completed-roster score rather than this display score.

Draft availability is modeled separately. Opponents draw from Sleeper PPR ADP,
with ECR as a labeled fallback when Sleeper has no market observation. Drafts
from both saved leagues are used when the same Sleeper managers appear. Other-
league observations receive 65% weight, older seasons decay by 0.70, and each
manager-position effect is shrunk by `n / (n + 20)` to keep small samples from
overriding platform ADP. The user's future simulated picks also receive a
capped tier-urgency adjustment based on conditional next-turn availability.

Sleeper does not project every custom scoring event. Defensive three-and-outs
are estimated from projected sacks; current injury labels and depth charts are
signals rather than medical forecasts. The season simulation also does not yet
model the real NFL schedule, coordinated bye weeks, waivers, trades, or lineup
decisions. Its playoff probability is a model estimate, not a sportsbook price.

The assistant polls Sleeper's read-only API. It never submits a draft pick, so
the recommended selection must still be made in Sleeper.

## Weekly lineup, waivers, and trades

Open **Manage team** from the draft screen, or visit `/team` after signing in.
The weekly tools work independently of draft order and use current ownership,
league scoring, starter slots, reserves, and taxi squads. Saved leagues and a
custom league ID/username are supported. A renewed league is followed only when
Sleeper supplies an explicit renewal chain.

```bash
uv run sleeper-draft --league shield-ai lineup
uv run sleeper-draft --league shield-ai waivers
uv run sleeper-draft --league shield-ai trades --limit 10
uv run sleeper-draft --league shield-ai lineup --week 2 --json
```

The default lineup maximizes expected weekly points using exact assignment,
including FLEX, SUPER_FLEX, REC_FLEX, WRRB_FLEX, and multiple-position eligibility.
Started players remain in their current slots, and started bench players stay on
the bench. Out/IR/suspended players and byes are excluded from new assignments.
Questionable players remain eligible with a reminder to check active status.
Missing projections remain missing; season totals are never substituted for
weekly points. Future lineup previews are allowed, while waiver/trade advice
uses the current week so current transaction locks remain meaningful.

**Fantasy Football Tiers** is an optional lineup preference. It matches the
site's STD, HALF, or PPR text lists to unambiguous Sleeper players. Positional
tiers guide within-position choices; projection-calibrated scores compare
positions in FLEX, with FLEX-specific lists used when available. Actual projected
points remain separate from the tiers preference score. Numerical upside is not
available from the text lists, so the tool does not invent an upside forecast.

The site currently omits season/week labels from its machine-readable pages.
The UI therefore asks you to check the linked charts and confirm the selected
season/week before using unlabeled tiers. Confirmation applies only to that
request; mismatched or clearly stale charts are rejected even when confirmed.
Equivalent CLI usage, after reviewing the site:

```bash
uv run sleeper-draft --league shield-ai lineup --mode tiers \
  --season 2026 --week 1 --confirm-tiers-week
```

Waivers screen the top 15 weekly projections and top 10 season-strength values
per position, then compare every eligible drop for each candidate. Open roster
slots require no drop. Anyone owned by another team, including reserve/taxi
players, is excluded. Suggestions show starter-point gain and depth gain after
the move, and respect disabled additions and active roster capacity. Claim
timing, FAAB, and final platform eligibility still need checking in Sleeper.

Trades check eligible one-for-one exchanges and a shortlist of 16 two-player
packages per team. Both teams must improve their modeled roster value, retain
starter coverage, and keep at least 90% of the outgoing package's season-strength
value. Equal-size packages avoid hidden extra drops. Disabled trading and the
league's trade deadline are respected. Balanced proposals are discussion
starters, not predictions that a manager will accept.

The move score combines starter projections, diminishing bench credit (12% of
the first backup and 4% of the second per position), and 2% of season strength.
Season strength is league-scored season projections divided by 17: an injury/bye
protection proxy, **not** a rest-of-season forecast or a market trade valuation.
Moves cannot discard more than 10% of known season strength; missing strength
estimates protect players from drops and trades. Keeper rights, picks, age, and
contracts are not valued. These bounded searches do not guarantee the best
possible season-long trade or waiver strategy.

Sources and freshness are displayed with each report. Weekly projections use a
15-minute cache; schedule data one minute; kickoff data five minutes. Raw caches
are scoped to source/season/week and rescored for each league. Expired weekly
data is not silently reused after an outage. Sleeper schedules and ESPN kickoff
times are cross-checked; missing kickoff times conservatively lock players with
games today or earlier. Unsupported scoring categories are disclosed.

## Draft strategy review

The review corrected rigid QB limits, incomplete flexible-slot handling,
late-board availability tracking, completed-draft handling, zero-projection
fallbacks, and missing positional reception premiums. Candidate comparisons use
paired random scenarios and account for the exact requested number of rollouts.
Simulated starter selection now uses information available before the sampled
outcome, removing hindsight that previously rewarded best-ball-like selection.
Regression tests cover the Python and rebuilt C++ paths.

Use `--engine auto` for unusual roster configurations: complex flex layouts,
future keepers, and unequal traded-pick layouts can require the Python engine.
The draft remains a projection-based heuristic with sampled future picks, not a
proof of the optimal draft. PPR-centric market ADP, historical-role assumptions,
bench weights, and simplified season simulations remain limitations. Weekly
management recommendations are separate from the simulated draft-season model.

## Phone interface on Cloud Run

The web service exposes the same native recommendations in a mobile layout. It
defaults to Shield AI After Hours, shows five recommendations, runs 15
candidates with 10,000 trials each, and polls Sleeper every two seconds. It is
read-only and protected by a signed login cookie.

Run it locally with a SHA-256 password digest and a random session secret:

```bash
APP_PASSWORD_HASH="$(printf %s 'choose-a-password' | sha256sum | cut -d' ' -f1)" \
SESSION_SECRET="$(openssl rand -hex 32)" COOKIE_SECURE=false \
uv run sleeper-draft-web
```

The production image compiles the C++23 engine and includes the newest local
fallback data in `.cache/`. The Cloud Run service should use one web worker,
request concurrency 1, four CPUs, at least 4 GiB of memory, and a request
timeout of at least 180 seconds. Keep the instance maximum at one so two phone
taps cannot create unexpected parallel simulation costs.

Rankings, projections, ADP, player metadata, nflverse statistics, and player-ID
mappings are cached under `.cache/`. The projection loader has a secondary
FFToday fallback, and each loader uses the newest compatible cache when its
online source is unavailable on draft night.

## Verification

```bash
uv run pytest -q
```
