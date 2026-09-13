from __future__ import annotations

import argparse
from statistics import median
from time import perf_counter

from sleeper_draft_assistant.models import LeagueContext, LeagueRules, Player
from sleeper_draft_assistant.native_engine import NativeMonteCarloDraft


def context() -> LeagueContext:
    teams = 10
    rounds = 15
    rules = LeagueRules(
        teams=teams,
        rounds=rounds,
        roster_positions=(
            "QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "FLEX",
            "K", "DEF", "BN", "BN", "BN", "BN", "BN",
        ),
        scoring={"rec": 1.0, "pass_td": 4.0},
    )
    return LeagueContext(
        league_id="benchmark",
        username="benchmark",
        user_id="user-1",
        roster_id=1,
        draft_slot=1,
        league={"name": "Native benchmark", "season": "2026"},
        draft={
            "draft_id": "benchmark",
            "draft_order": {f"user-{slot}": slot for slot in range(1, teams + 1)},
            "slot_to_roster_id": {str(slot): slot for slot in range(1, teams + 1)},
            "settings": {"teams": teams, "rounds": rounds},
        },
        picks=[],
        traded_picks=[],
        users=[],
        rosters=[],
        rules=rules,
    )


def players(count: int = 471) -> list[Player]:
    positions = ("RB", "WR", "RB", "WR", "QB", "TE", "K", "DEF")
    result = []
    for index in range(count):
        rank = float(index + 1)
        position = positions[index % len(positions)]
        baseline = {
            "QB": 320.0,
            "RB": 310.0,
            "WR": 305.0,
            "TE": 250.0,
            "K": 125.0,
            "DEF": 110.0,
        }[position]
        result.append(
            Player(
                sleeper_id=f"player-{index}",
                name=f"Player {index}",
                position=position,
                team="TST",
                ecr=rank,
                uncertainty=8.0,
                projected_points=max(20.0, baseline - index * 0.55),
                value_rank=rank,
                adp=rank,
                adp_uncertainty=10.0,
                outcome_cv=0.25,
            )
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=int, default=15)
    parser.add_argument("--runs-per-candidate", type=int, default=10_000)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--max-seconds", type=float, default=60.0)
    args = parser.parse_args()

    engine = NativeMonteCarloDraft(context(), players(), seed=2026)
    engine.recommend(1_500, args.candidates, "auto")
    total = args.candidates * args.runs_per_candidate
    timings = []
    for trial in range(1, args.trials + 1):
        started = perf_counter()
        report = engine.recommend(total, args.candidates, "auto")
        elapsed = perf_counter() - started
        timings.append(elapsed)
        print(
            f"trial={trial} rollouts={report.total_rollouts:,} "
            f"elapsed={elapsed:.3f}s rate={report.total_rollouts / elapsed:,.0f}/s"
        )

    typical = median(timings)
    slowest = max(timings)
    print(f"median={typical:.3f}s max={slowest:.3f}s target={args.max_seconds:.3f}s")
    return int(typical > 55.0 or slowest > args.max_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
