from __future__ import annotations

from sleeper_draft_assistant.models import LeagueContext, LeagueRules, Player
from sleeper_draft_assistant.simulator import (
    MonteCarloDraft,
    roster_for_pick,
    snake_slot_for_pick,
)


def make_context(
    *,
    teams: int = 4,
    rounds: int = 4,
    positions: tuple[str, ...] = ("QB", "RB", "WR", "FLEX"),
) -> LeagueContext:
    draft_order = {f"user-{slot}": slot for slot in range(1, teams + 1)}
    slot_to_roster = {str(slot): slot for slot in range(1, teams + 1)}
    rules = LeagueRules(
        teams=teams,
        rounds=rounds,
        roster_positions=positions,
        scoring={"rec": 1.0, "pass_td": 6.0},
    )
    return LeagueContext(
        league_id="league",
        username="user-1",
        user_id="user-1",
        roster_id=1,
        draft_slot=1,
        league={
            "name": "Test",
            "season": "2026",
            "previous_league_id": None,
            "settings": {
                "playoff_teams": 2,
                "start_week": 1,
                "playoff_week_start": 5,
            },
        },
        draft={
            "draft_id": "draft",
            "draft_order": draft_order,
            "slot_to_roster_id": slot_to_roster,
            "settings": {"teams": teams, "rounds": rounds},
        },
        picks=[],
        traded_picks=[],
        users=[],
        rosters=[],
        rules=rules,
    )


def make_players(count_per_position: int = 8) -> list[Player]:
    players = []
    overall_rank = 1
    for position in ("RB", "WR", "QB", "TE"):
        for index in range(count_per_position):
            players.append(
                Player(
                    sleeper_id=f"{position}-{index}",
                    name=f"{position} Player {index}",
                    position=position,
                    team="TST",
                    ecr=float(overall_rank),
                    uncertainty=2.0,
                )
            )
            overall_rank += 1
    return players


def test_snake_slots_reverse_in_even_rounds() -> None:
    assert [snake_slot_for_pick(pick, 4) for pick in range(1, 9)] == [
        1,
        2,
        3,
        4,
        4,
        3,
        2,
        1,
    ]


def test_traded_pick_changes_roster_owner() -> None:
    context = make_context()
    context.traded_picks = [{"round": 2, "roster_id": 4, "owner_id": 2}]
    assert roster_for_pick(context, 5) == 2


def test_recommendation_runs_from_an_empty_board() -> None:
    simulator = MonteCarloDraft(make_context(), make_players(), seed=7)
    report = simulator.recommend(simulations=60, candidate_count=3)

    assert report.on_clock
    assert report.next_user_pick == 1
    assert len(report.recommendations) == 3
    assert report.total_rollouts >= 75
    assert all(item.samples >= 25 for item in report.recommendations)


def test_existing_picks_advance_the_board() -> None:
    context = make_context()
    context.picks = [
        {"pick_no": 1, "roster_id": 1, "player_id": "RB-0", "metadata": {"position": "RB"}},
        {"pick_no": 2, "roster_id": 2, "player_id": "RB-1", "metadata": {"position": "RB"}},
    ]
    simulator = MonteCarloDraft(context, make_players(), seed=9)

    assert simulator.base_state.next_pick == 3
    assert simulator.base_state.rosters[1] == ["RB-0"]
    assert simulator.base_state.drafted == {"RB-0", "RB-1"}


def test_analysis_summarizes_drafts_and_playoff_results() -> None:
    simulator = MonteCarloDraft(make_context(), make_players(), seed=17)
    report = simulator.analyze(simulations=40, weekly_variance=0.14, top_players=6)

    assert report.simulations == 40
    assert report.regular_season_weeks == 4
    assert report.playoff_teams == 2
    assert 0.0 <= report.playoff_rate <= 1.0
    assert 0.0 <= report.top_two_rate <= 1.0
    assert sum(report.finish_rates) == 1.0
    assert len(report.finish_rates) == 4
    assert len(report.common_players) == 6
    assert len(report.representative_run.picks) == 4
    assert 0.0 <= report.average_wins <= report.regular_season_weeks


def test_round_robin_pairs_every_team_once_per_round() -> None:
    rounds = MonteCarloDraft._round_robin([1, 2, 3, 4])

    assert len(rounds) == 3
    assert all(sorted(team for pair in week for team in pair) == [1, 2, 3, 4] for week in rounds)
    assert {tuple(sorted(pair)) for week in rounds for pair in week} == {
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    }
