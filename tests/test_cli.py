from __future__ import annotations

from sleeper_draft_assistant.cli import build_parser


def test_workers_argument_accepts_integer() -> None:
    args = build_parser().parse_args(["recommend", "--workers", "4"])

    assert args.workers == 4


def test_workers_argument_accepts_auto() -> None:
    args = build_parser().parse_args(["watch", "--workers", "auto"])

    assert args.workers == "auto"
