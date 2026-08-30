# Sleeper Draft Assistant

Monte Carlo draft recommendations tailored to a Sleeper league's scoring,
roster slots, draft order, completed picks, and prior manager tendencies.

The defaults are configured for `brycexander` in The Unemployables
(`1387590026778411008`), with Hooligans (`1389738046894657536`) available as a
saved league. Player value combines Sleeper season projections,
Sleeper PPR ADP, and the current full-PPR Expert Consensus Ranking board loaded
through `nflreadpy`.

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

Run the same league-aware advice for Hooligans:

```bash
uv run sleeper-draft --league hooligans recommend
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

## Model Inputs

Player quality starts with raw Sleeper season stat projections scored against
the league's actual scoring settings. It then calculates value over replacement
for this league's eight teams, required starters, and two flex spots. That VORP
signal receives 82% of the composite weight; ECR is an 18% supporting prior.
Current injury designation, depth-chart role, rookie status, age, and expert
disagreement affect the risk penalty or the width of simulated outcomes.

Draft availability is modeled separately. Opponents draw from Sleeper PPR ADP,
with ECR as a labeled fallback when Sleeper has no market observation. Prior
league drafts shift each manager's position timing. This separation lets the
assistant identify a valuable player without assuming that player must be taken
well ahead of the market.

Sleeper does not project every custom scoring event. Defensive three-and-outs
are estimated from projected sacks; current injury labels and depth charts are
signals rather than medical forecasts. The season simulation also does not yet
model the real NFL schedule, coordinated bye weeks, waivers, trades, or lineup
decisions. Its playoff probability is a model estimate, not a sportsbook price.

The assistant polls Sleeper's read-only API. It never submits a draft pick, so
the recommended selection must still be made in Sleeper.

Rankings, projections, ADP, and player metadata are cached by date under
`.cache/`. The projection loader has a secondary FFToday fallback, and the
ranking loader uses the newest cache if a refresh fails on draft night.

## Verification

```bash
uv run pytest -q
```
