from __future__ import annotations

import pytest

from sleeper_draft_assistant.cli import (
    LEAGUE_IDS,
    _league_id,
    _simulation_total,
    build_parser,
)


def test_default_league_is_unemployables() -> None:
    args = build_parser().parse_args(["league"])

    assert _league_id(args) == LEAGUE_IDS["unemployables"]


def test_hooligans_league_preset() -> None:
    args = build_parser().parse_args(["--league", "hooligans", "recommend"])

    assert _league_id(args) == "1389738046894657536"


def test_shield_ai_league_preset() -> None:
    args = build_parser().parse_args(["--league", "shield-ai", "recommend"])

    assert _league_id(args) == "1401668384990494720"


def test_explicit_league_id_is_still_supported() -> None:
    args = build_parser().parse_args(["--league-id", "123", "league"])

    assert _league_id(args) == "123"


def test_workers_argument_accepts_integer() -> None:
    args = build_parser().parse_args(["recommend", "--workers", "4"])

    assert args.workers == 4


def test_workers_argument_accepts_auto() -> None:
    args = build_parser().parse_args(["watch", "--workers", "auto"])

    assert args.workers == "auto"


def test_runs_per_candidate_sets_exact_total() -> None:
    args = build_parser().parse_args(
        ["recommend", "--candidates", "15", "--runs-per-candidate", "10000"]
    )

    assert _simulation_total(args) == 150_000
    assert args.engine == "auto"


def test_run_count_options_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["recommend", "--simulations", "100", "--runs-per-candidate", "10"]
        )
