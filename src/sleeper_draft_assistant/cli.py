from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time

from .models import LeagueContext, Player
from .rankings import load_consensus_board
from .simulator import DraftAnalysisReport, MonteCarloDraft, SimulationReport
from .sleeper import SleeperClient
from .valuation import ValuationDiagnostics, build_player_values


DEFAULT_USERNAME = "brycexander"
DEFAULT_LEAGUE = "unemployables"
LEAGUE_IDS = {
    "unemployables": "1387590026778411008",
    "hooligans": "1389738046894657536",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sleeper-draft",
        description="League-aware Monte Carlo recommendations for a Sleeper draft.",
    )
    league = parser.add_mutually_exclusive_group()
    league.add_argument(
        "--league",
        choices=tuple(LEAGUE_IDS),
        default=DEFAULT_LEAGUE,
        help=f"Saved league to use (default: {DEFAULT_LEAGUE})",
    )
    league.add_argument("--league-id", help="Sleeper league ID (overrides saved leagues)")
    parser.add_argument("--username", default=DEFAULT_USERNAME)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("league", help="Show the synced league and draft configuration")

    values = subparsers.add_parser(
        "values", help="Show the league-adjusted player value board"
    )
    values.add_argument("--top", type=int, default=40, help="Number of players to show")
    values.add_argument(
        "--position", choices=("QB", "RB", "WR", "TE", "K", "DEF")
    )
    values.add_argument("--refresh-rankings", action="store_true")

    recommend = subparsers.add_parser("recommend", help="Simulate and recommend the next pick")
    _add_simulation_arguments(recommend)

    analyze = subparsers.add_parser(
        "analyze", help="Summarize draft outcomes and estimate playoff probability"
    )
    _add_simulation_arguments(analyze, include_candidates=False)
    analyze.add_argument(
        "--weekly-variance",
        type=float,
        default=0.22,
        help="Week-to-week team score variation (default: 0.22)",
    )
    analyze.add_argument(
        "--top-players", type=int, default=15, help="Number of common players to show"
    )

    watch = subparsers.add_parser("watch", help="Watch the draft and rerun after each pick")
    _add_simulation_arguments(watch)
    watch.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds")
    return parser


def _add_simulation_arguments(
    parser: argparse.ArgumentParser, include_candidates: bool = True
) -> None:
    parser.add_argument(
        "--simulations", type=int, default=1000, help="Total Monte Carlo rollouts"
    )
    if include_candidates:
        parser.add_argument(
            "--candidates", type=int, default=10, help="Options tested on the clock"
        )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--workers",
        default=1,
        type=_parse_workers,
        help="Parallel worker processes for simulations, or 'auto' (default: 1)",
    )
    parser.add_argument(
        "--history-seasons",
        type=int,
        default=3,
        help="Previous league seasons used for manager tendencies",
    )
    parser.add_argument("--refresh-rankings", action="store_true")


def _parse_workers(value: str) -> int | str:
    if value == "auto":
        return value
    workers = int(value)
    if workers < 1:
        raise argparse.ArgumentTypeError("--workers must be >= 1 or 'auto'")
    return workers


def _worker_label(workers: int | str) -> str:
    if workers == "auto":
        return f"auto ({max(1, (os.cpu_count() or 2) - 1)})"
    return str(workers)


def _league_id(args: argparse.Namespace) -> str:
    return args.league_id or LEAGUE_IDS[args.league]


def _print_league(context: LeagueContext) -> None:
    start_ms = context.draft.get("start_time")
    start = (
        datetime.fromtimestamp(float(start_ms) / 1000).astimezone().strftime("%a %b %-d, %Y %-I:%M %p %Z")
        if start_ms
        else "not scheduled"
    )
    scoring = context.rules.scoring
    print(f"{context.league['name'].strip()} ({context.league['season']})")
    print(f"League ID: {context.league_id}")
    print(f"Draft: {context.draft['type']} | {context.rules.teams} teams | {context.rules.rounds} rounds")
    print(f"Starts: {start}")
    print(f"Your slot: {context.draft_slot} | roster ID: {context.roster_id}")
    print(f"Roster: {' '.join(context.rules.roster_positions)}")
    print(
        "Scoring: "
        f"{scoring.get('rec', 0):g} PPR | "
        f"{scoring.get('pass_td', 0):g}-point passing TD | "
        f"{scoring.get('pass_int', 0):g} interception"
    )
    print(f"Completed picks: {len(context.picks)}")
    print(f"Keepers: {sum(len(roster.get('keepers') or []) for roster in context.rosters)}")
    print(f"Traded picks: {len(context.traded_picks)}")


def _print_report(context: LeagueContext, report: SimulationReport) -> None:
    round_number = (report.next_user_pick - 1) // context.rules.teams + 1
    pick_in_round = (report.next_user_pick - 1) % context.rules.teams + 1
    status = "ON THE CLOCK" if report.on_clock else f"next selection: {round_number}.{pick_in_round:02d}"
    print()
    print(f"Monte Carlo result ({report.total_rollouts:,} rollouts, {status})")
    if report.on_clock:
        print("Rank  Player                    Pos Team  Top roster  Model score  Sims")
        print("----  ------------------------  --- ----  ----------  -----------  ----")
        for rank, item in enumerate(report.recommendations, start=1):
            print(
                f"{rank:>4}  {item.player.name[:24]:<24}  {item.player.position:<3} "
                f"{item.player.team[:4]:<4}  {item.top_roster_rate:>9.1%}  "
                f"{item.mean_score:>11.1f}  {item.samples:>4}"
            )
    else:
        print("Player                    Pos Team  Available  AI selects  Top roster  Sims")
        print("------------------------  --- ----  ---------  ----------  ----------  ----")
        for item in report.recommendations:
            print(
                f"{item.player.name[:24]:<24}  {item.player.position:<3} {item.player.team[:4]:<4}  "
                f"{item.availability_rate:>8.1%}  {item.selection_rate:>9.1%}  "
                f"{item.top_roster_rate:>9.1%}  {item.samples:>4}"
            )
        print("AI selects is the adaptive recommendation after the picks ahead of you are simulated.")
    print(
        "Top roster uses league-scored projections and uncertain player outcomes; "
        "it is not a literal championship probability."
    )


def _print_analysis(context: LeagueContext, report: DraftAnalysisReport) -> None:
    baseline = report.playoff_teams / context.rules.teams
    edge = report.playoff_rate - baseline
    sampling_margin = 1.96 * (
        report.playoff_rate * (1.0 - report.playoff_rate) / report.simulations
    ) ** 0.5
    average_losses = report.regular_season_weeks - report.average_wins
    print()
    print(f"Draft and season analysis ({report.simulations:,} simulations)")
    print(
        f"Estimated playoff probability: {report.playoff_rate:.1%} "
        f"(sampling margin +/-{sampling_margin:.1%})"
    )
    print(f"Equal-team baseline:            {baseline:.1%} ({edge:+.1%} model edge)")
    print(f"Top-two finish probability:     {report.top_two_rate:.1%}")
    print(f"Average finish:                 {report.average_finish:.2f} of {context.rules.teams}")
    print(
        f"Average regular-season record:  {report.average_wins:.1f}-"
        f"{average_losses:.1f} over {report.regular_season_weeks} weeks"
    )

    print("\nFinish distribution")
    for finish, rate in enumerate(report.finish_rates, start=1):
        bar = "#" * max(1, round(rate * 30)) if rate else ""
        print(f"{finish:>2}: {rate:>6.1%}  {bar}")

    print("\nMost common players on your simulated teams")
    print("Player                    Pos Team  Rostered  Avg round")
    print("------------------------  --- ----  --------  ---------")
    for item in report.common_players:
        print(
            f"{item.player.name[:24]:<24}  {item.player.position:<3} "
            f"{item.player.team[:4]:<4}  {item.roster_rate:>7.1%}  "
            f"{item.average_round:>9.2f}"
        )

    print("\nMost common four-pick openings")
    for opening, rate in report.common_openings:
        print(f"{rate:>6.1%}  {'-'.join(opening)}")

    print("\nMost common final roster builds")
    position_order = ("QB", "RB", "WR", "TE", "K", "DEF")
    for build, rate in report.common_builds:
        counts = dict(build)
        label = " ".join(
            f"{counts[position]} {position}" for position in position_order if counts.get(position)
        )
        print(f"{rate:>6.1%}  {label}")

    run = report.representative_run
    print(
        f"\nRepresentative median run: {run.finish} place, "
        f"{run.wins:g}-{report.regular_season_weeks - run.wins:g}"
    )
    print("Round  Player                    Pos Team")
    print("-----  ------------------------  --- ----")
    for round_number, player in run.picks:
        print(
            f"{round_number:>5}  {player.name[:24]:<24}  "
            f"{player.position:<3} {player.team[:4]:<4}"
        )
    print(
        "\nPlayoff probability is based on projected player production, uncertain outcomes, "
        "a balanced schedule, and weekly variance; it is not a sportsbook forecast."
    )


def _print_valuation_diagnostics(diagnostics: ValuationDiagnostics) -> None:
    projection_sources = ", ".join(
        f"{source} {count}"
        for source, count in sorted(diagnostics.projection_counts.items())
    )
    adp_sources = ", ".join(
        f"{source} {count}" for source, count in sorted(diagnostics.adp_counts.items())
    )
    replacement = ", ".join(
        f"{position} {points:.1f}"
        for position, points in diagnostics.replacement_points.items()
    )
    print(f"Projection coverage: {projection_sources}")
    print(f"Draft-market coverage: {adp_sources}")
    print(f"Replacement baselines: {replacement} season points")


def _load_values(
    context: LeagueContext,
    args: argparse.Namespace,
) -> tuple[list[Player], ValuationDiagnostics]:
    cache_dir = Path(__file__).resolve().parents[2] / ".cache"
    print("Loading current full-PPR expert consensus rankings...")
    board = load_consensus_board(cache_dir, refresh=args.refresh_rankings)
    print("Scoring projections and replacement value for this league...")
    result = build_player_values(
        board,
        context.rules,
        cache_dir,
        int(context.league["season"]),
        refresh=args.refresh_rankings,
    )
    _print_valuation_diagnostics(result.diagnostics)
    return list(result.players), result.diagnostics


def _print_values(players: list[Player], args: argparse.Namespace) -> None:
    if args.position:
        players = [player for player in players if player.position == args.position]
    players = players[: max(1, args.top)]
    print()
    print("Value  Player                    Pos Team   Proj   VORP   ECR    ADP  Inj  Role  Source")
    print("-----  ------------------------  --- ----  -----  -----  -----  -----  ---  ----  -------------")
    for player in players:
        adp = f"{player.adp:.1f}" if player.adp is not None else "-"
        print(
            f"{player.value_rank:>5.0f}  {player.name[:24]:<24}  {player.position:<3} "
            f"{player.team[:4]:<4}  {player.projected_points:>5.1f}  "
            f"{player.vorp:>5.1f}  {player.ecr:>5.1f}  {adp:>5}  "
            f"{player.injury_risk:>3.0%}  {player.role_risk:>4.0%}  "
            f"{player.projection_source}"
        )
    print(
        "\nValue blends league-specific VORP (82%) with ECR (18%), then applies "
        "current injury and role-risk penalties. ADP predicts availability, not quality."
    )


def _load_simulator(
    client: SleeperClient,
    context: LeagueContext,
    args: argparse.Namespace,
    board=None,
    biases=None,
) -> tuple[MonteCarloDraft, list, dict]:
    if board is None:
        board, _ = _load_values(context, args)
    if biases is None:
        print(f"Loading up to {args.history_seasons} prior season(s) of manager tendencies...")
        biases = client.manager_position_biases(context.league, args.history_seasons)
    simulator = MonteCarloDraft(context, board, biases, seed=args.seed)
    return simulator, board, biases


def _recommend(client: SleeperClient, context: LeagueContext, args: argparse.Namespace) -> None:
    simulator, _, _ = _load_simulator(client, context, args)
    report = simulator.recommend(args.simulations, args.candidates, args.workers)
    _print_report(context, report)


def _analyze(client: SleeperClient, context: LeagueContext, args: argparse.Namespace) -> None:
    simulator, _, _ = _load_simulator(client, context, args)
    report = simulator.analyze(
        simulations=args.simulations,
        weekly_variance=args.weekly_variance,
        top_players=args.top_players,
        workers=args.workers,
    )
    _print_analysis(context, report)


def _watch(client: SleeperClient, context: LeagueContext, args: argparse.Namespace) -> None:
    _, board, biases = _load_simulator(client, context, args)
    draft_id = str(context.draft["draft_id"])
    last_signature: tuple[tuple[int, str], ...] | None = None
    print(
        f"Watching draft every {args.interval:g}s with {_worker_label(args.workers)} "
        "worker(s). Press Ctrl-C to stop."
    )

    while True:
        picks = client.picks(draft_id)
        signature = tuple(
            (int(pick["pick_no"]), str(pick["player_id"])) for pick in picks
        )
        if signature != last_signature:
            context.picks = picks
            simulator = MonteCarloDraft(context, board, biases, seed=args.seed + len(picks))
            print(f"\nBoard updated: {len(picks)} pick(s) complete")
            _print_report(
                context,
                simulator.recommend(args.simulations, args.candidates, args.workers),
            )
            last_signature = signature
        time.sleep(args.interval)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = SleeperClient()
    try:
        context = client.sync(_league_id(args), args.username)
        _print_league(context)
        if args.command == "league":
            return 0
        if args.command == "values":
            players, _ = _load_values(context, args)
            _print_values(players, args)
            return 0
        if args.command == "recommend":
            _recommend(client, context, args)
            return 0
        if args.command == "analyze":
            _analyze(client, context, args)
            return 0
        if args.command == "watch":
            _watch(client, context, args)
            return 0
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
