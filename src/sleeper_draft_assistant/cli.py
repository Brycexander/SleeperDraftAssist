from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

from .cache import cache_directory
from .models import LeagueContext, Player
from .native_engine import create_recommendation_engine
from .rankings import load_consensus_board
from .simulator import DraftAnalysisReport, MonteCarloDraft, SimulationReport
from .sleeper import SleeperClient
from .valuation import ValuationDiagnostics, build_player_values
from .expert_sources import import_expert_snapshot


DEFAULT_USERNAME = "brycexander"
DEFAULT_LEAGUE = "unemployables"
LEAGUE_IDS = {
    "unemployables": "1387590026778411008",
    "hooligans": "1389738046894657536",
    "shield-ai": "1401668384990494720",
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
    for command, help_text in (
        ("lineup", "Pick the strongest legal weekly starting lineup"),
        ("waivers", "Find unrostered players that improve your team"),
        ("trades", "Find balanced trades that improve both teams"),
    ):
        team = subparsers.add_parser(command, help=help_text)
        team.add_argument("--week", type=int, choices=range(1, 19), help="NFL week (default: current)")
        team.add_argument("--season", type=int, help="Verify the current season, required to confirm unlabeled tiers")
        team.add_argument("--mode", choices=("projection", "tiers"), default="projection")
        team.add_argument("--limit", type=_positive_int, default=10)
        team.add_argument("--refresh", action="store_true", help="Refresh weekly source data")
        team.add_argument("--confirm-tiers-week", action="store_true", help="Confirm you checked the tiers site matches --season and --week")
        team.add_argument("--json", action="store_true", help="Print the full structured report")
        if command in {"waivers", "trades"}:
            team.add_argument("--valuation-source", choices=("sleeper", "experts"), default="sleeper",
                              help="Use legacy Sleeper projections or imported expert ROS ranks")
        if command == "trades":
            team.add_argument("--trade-approach", choices=("balanced", "opportunities"), default="balanced",
                              help="Balanced swaps or speculative buy-low/sell-high ideas")
    import_cmd = subparsers.add_parser("import-expert-rankings", help="Import a validated expert ROS JSON snapshot")
    import_cmd.add_argument("path", type=Path)
    import_cmd.add_argument("--week", type=int, choices=range(1, 19), required=True)
    import_cmd.add_argument("--scoring", choices=("STD", "HALF", "PPR"), required=True)
    return parser


def _add_simulation_arguments(
    parser: argparse.ArgumentParser, include_candidates: bool = True
) -> None:
    simulations = parser.add_mutually_exclusive_group()
    simulations.add_argument(
        "--simulations", type=_positive_int, help="Total Monte Carlo rollouts (default: 1000)"
    )
    if include_candidates:
        parser.add_argument(
            "--candidates", type=_positive_int, default=10, help="Options tested on the clock"
        )
        simulations.add_argument(
            "--runs-per-candidate",
            type=_positive_int,
            help="Rollouts per candidate; total rollouts are this value times --candidates",
        )
        parser.add_argument(
            "--engine",
            choices=("auto", "cpp", "python"),
            default="auto",
            help="Recommendation engine (default: auto)",
        )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--workers",
        default=1,
        type=_parse_workers,
        help="Native threads or Python processes, or 'auto' (default: 1)",
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


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return number


def _simulation_total(args: argparse.Namespace) -> int:
    runs_per_candidate = getattr(args, "runs_per_candidate", None)
    if runs_per_candidate is not None:
        return runs_per_candidate * args.candidates
    return args.simulations if args.simulations is not None else 1000


def _worker_label(workers: int | str, engine_name: str = "python") -> str:
    if workers == "auto":
        detected = os.cpu_count() or 2
        count = detected if engine_name == "cpp" else max(1, detected - 1)
        return f"auto ({count})"
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
        print(
            "Rank  Player                    Pos Team  Lineup +  Next turn  "
            "Wait cost  Top roster  Model score  Sims"
        )
        print(
            "----  ------------------------  --- ----  --------  ---------  "
            "---------  ----------  -----------  ----"
        )
        for rank, item in enumerate(report.recommendations, start=1):
            next_turn = (
                f"{item.next_pick_availability:.1%}"
                if item.next_pick_availability is not None
                else "n/a"
            )
            print(
                f"{rank:>4}  {item.player.name[:24]:<24}  {item.player.position:<3} "
                f"{item.player.team[:4]:<4}  {item.starter_gain:>8.1f}  "
                f"{next_turn:>9}  {item.wait_cost:>9.1f}  "
                f"{item.top_roster_rate:>9.1%}  "
                f"{item.mean_score:>11.1f}  {item.samples:>4}"
            )
    else:
        print(
            "Player                    Pos Team  Available  AI selects  Lineup +  "
            "Top roster  Sims"
        )
        print(
            "------------------------  --- ----  ---------  ----------  --------  "
            "----------  ----"
        )
        for item in report.recommendations:
            print(
                f"{item.player.name[:24]:<24}  {item.player.position:<3} {item.player.team[:4]:<4}  "
                f"{item.availability_rate:>8.1%}  {item.selection_rate:>9.1%}  "
                f"{item.starter_gain:>8.1f}  "
                f"{item.top_roster_rate:>9.1%}  {item.samples:>4}"
            )
        print("AI selects is the adaptive recommendation after the picks ahead of you are simulated.")
    print(
        "Top roster uses league-scored projections and uncertain player outcomes; "
        "it is not a literal championship probability."
    )
    if report.on_clock:
        print(
            "Lineup + is the projected starter/bench gain. Next turn is the "
            "conditional chance the player survives; wait cost is expected VORP lost."
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
    if diagnostics.history_seasons:
        seasons = "-".join(str(season) for season in diagnostics.history_seasons)
        print(
            f"Historical calibration: {diagnostics.historical_players} players "
            f"with usable {seasons} results"
        )
    print(f"Draft-market coverage: {adp_sources}")
    print(f"Replacement baselines: {replacement} season points")


def _load_values(
    context: LeagueContext,
    args: argparse.Namespace,
) -> tuple[list[Player], ValuationDiagnostics]:
    cache_dir = cache_directory()
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
    print(
        "Rank  Player                    Pos Team  Score  ProjV  ECRV  Risk  "
        "Online   Hist   H%  Blend   VORP    ADP"
    )
    print(
        "----  ------------------------  --- ----  -----  -----  ----  ----  "
        "------  -----  ---  -----  -----  -----"
    )
    for player in players:
        adp = f"{player.adp:.1f}" if player.adp is not None else "-"
        historical = (
            f"{player.historical_points:.1f}"
            if player.history_weight > 0
            else "-"
        )
        print(
            f"{player.value_rank:>5.0f}  {player.name[:24]:<24}  {player.position:<3} "
            f"{player.team[:4]:<4}  {player.value_score:>5.1f}  "
            f"{player.projection_component:>5.1f}  {player.consensus_component:>4.1f}  "
            f"{player.risk_penalty:>4.1f}  {player.online_projected_points:>6.1f}  "
            f"{historical:>5}  {player.history_weight:>3.0%}  "
            f"{player.projected_points:>5.1f}  {player.vorp:>5.1f}  {adp:>5}"
        )
    print(
        "\nBlend combines current online projections with a reliability-weighted "
        "historical baseline (maximum 10-25% by position). Score = ProjV + ECRV "
        "- Risk. ADP remains an availability signal, not player quality."
    )


def _load_simulator(
    client: SleeperClient,
    context: LeagueContext,
    args: argparse.Namespace,
    board=None,
    biases=None,
) -> tuple[MonteCarloDraft | object, list, dict, str]:
    if board is None:
        board, _ = _load_values(context, args)
    if biases is None:
        print(f"Loading up to {args.history_seasons} prior season(s) of manager tendencies...")
        biases = client.manager_position_biases(
            context.league,
            args.history_seasons,
            additional_league_ids=LEAGUE_IDS.values(),
        )
    simulator, engine_name = create_recommendation_engine(
        args.engine, context, board, biases, seed=args.seed
    )
    return simulator, board, biases, engine_name


def _recommend(client: SleeperClient, context: LeagueContext, args: argparse.Namespace) -> None:
    started = time.perf_counter()
    simulator, _, _, engine_name = _load_simulator(client, context, args)
    prepared = time.perf_counter()
    total = _simulation_total(args)
    report = simulator.recommend(total, args.candidates, args.workers)
    simulated = time.perf_counter()
    _print_report(context, report)
    per_candidate = (
        f" | requested per candidate {args.runs_per_candidate:,}"
        if args.runs_per_candidate is not None
        else ""
    )
    print(
        f"Timing: engine {engine_name} | workers {_worker_label(args.workers, engine_name)}{per_candidate} | "
        f"model preparation {prepared - started:.1f}s | "
        f"simulations {simulated - prepared:.1f}s | total {simulated - started:.1f}s"
    )


def _analyze(client: SleeperClient, context: LeagueContext, args: argparse.Namespace) -> None:
    board, _ = _load_values(context, args)
    print(f"Loading up to {args.history_seasons} prior season(s) of manager tendencies...")
    biases = client.manager_position_biases(
        context.league,
        args.history_seasons,
        additional_league_ids=LEAGUE_IDS.values(),
    )
    simulator = MonteCarloDraft(context, board, biases, seed=args.seed)
    report = simulator.analyze(
        simulations=_simulation_total(args),
        weekly_variance=args.weekly_variance,
        top_players=args.top_players,
        workers=args.workers,
    )
    _print_analysis(context, report)


def _watch(client: SleeperClient, context: LeagueContext, args: argparse.Namespace) -> None:
    simulator, board, biases, engine_name = _load_simulator(client, context, args)
    draft_id = str(context.draft["draft_id"])
    last_signature: tuple[tuple[int, str], ...] | None = None
    total = _simulation_total(args)
    print(
        f"Watching draft every {args.interval:g}s with {engine_name} engine and "
        f"{_worker_label(args.workers, engine_name)} worker(s), {total:,} rollouts. "
        "Press Ctrl-C to stop."
    )

    while True:
        picks = client.picks(draft_id)
        signature = tuple(
            (int(pick["pick_no"]), str(pick["player_id"])) for pick in picks
        )
        if signature != last_signature:
            started = time.perf_counter()
            context.picks = picks
            if hasattr(simulator, "update_picks"):
                simulator.update_picks(picks, args.seed + len(picks))
            else:
                simulator = MonteCarloDraft(
                    context, board, biases, seed=args.seed + len(picks)
                )
            prepared = time.perf_counter()
            print(f"\nBoard updated: {len(picks)} pick(s) complete")
            _print_report(
                context,
                simulator.recommend(total, args.candidates, args.workers),
            )
            simulated = time.perf_counter()
            per_candidate = (
                f" | requested per candidate {args.runs_per_candidate:,}"
                if args.runs_per_candidate is not None
                else ""
            )
            print(
                f"Timing: engine {engine_name} | workers "
                f"{_worker_label(args.workers, engine_name)}{per_candidate} | "
                f"model preparation {prepared - started:.1f}s | "
                f"simulations {simulated - prepared:.1f}s | "
                f"total {simulated - started:.1f}s"
            )
            last_signature = signature
        time.sleep(args.interval)


def _manage_team(client: SleeperClient, args: argparse.Namespace) -> None:
    from .management import TeamService

    report = TeamService(client).advise(
        _league_id(args), args.username, action=args.command, week=args.week,
        mode=args.mode, limit=args.limit, refresh=args.refresh,
        confirm_tiers_week=args.confirm_tiers_week, expected_season=args.season,
        trade_approach=getattr(args, "trade_approach", "balanced"),
        valuation_source=getattr(args, "valuation_source", "sleeper"),
    )
    if args.json:
        print(json.dumps(report, indent=2, allow_nan=False))
        return
    print(f"{report['league_name']} | {report['season']} week {report['week']} | {report['scoring']['ppr']:g} PPR")
    print(report["method"])
    lineup = report["lineup"]
    if lineup:
        print(f"Projected starters: {lineup['projected_points']:.2f} points")
        for row in lineup["starters"]:
            points = "no projection" if row.get("points") is None else f"{row['points']:.2f} pts"
            print(f"  {row['slot']:12s} {row.get('name') or 'Empty slot':28s} {points}")
    for item in report["suggestions"]:
        if args.command == "waivers":
            drop = item.get("drop")
            gain = item.get("weekly_gain", item.get("own_gain", {}).get("starters", 0.0))
            print(f"Add {item['add']['name']} / drop {drop['name'] if drop else 'nobody (open slot)'}: {gain:+.2f} rank-credit starters")
        else:
            give = ", ".join(p["name"] for p in item["give"])
            receive = ", ".join(p["name"] for p in item["receive"])
            own_gain = item.get("own_gain", {}).get("starters", item.get("weekly_gain", 0.0))
            partner_gain = item.get("partner_gain", {}).get("starters", item.get("partner_weekly_gain", 0.0))
            print(f"Team {item['partner_roster_id']}: send {give}; receive {receive}. You {own_gain:+.2f}, partner {partner_gain:+.2f} starter rank credits")
        print(f"  {item['reason']}")
        if item.get("kind") == "buy_low_sell_high":
            print(f"  Outlook gain: {item['outlook_gain']:+.2f}; partner outlook {item['partner_outlook_gain']:+.2f}; partner perceived gain {item['partner_perceived_gain']:+.2f}")
            print(f"  Possible discussion opener: {item['pitch']}")
        for warning in item.get("warnings", []):
            print(f"  Note: {warning}")
    if args.command != "lineup" and not report["suggestions"]:
        print("No improving moves met the data and balance requirements.")
    for warning in report["warnings"]:
        print(f"Note: {warning}")
    for source in report["sources"]:
        print(f"Source: {json.dumps(source)}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = SleeperClient()
    try:
        if args.command == "import-expert-rankings":
            state = client.nfl_state()
            result = import_expert_snapshot(args.path.read_bytes(), cache_directory(), season=int(state["season"]),
                                            week=args.week, scoring=args.scoring, now=datetime.now().astimezone())
            print(json.dumps(result, indent=2))
            return 0 if result.get("status") == "ready" else 1
        if args.command in {"lineup", "waivers", "trades"}:
            _manage_team(client, args)
            return 0
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
