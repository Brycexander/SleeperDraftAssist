from dataclasses import replace
from sleeper_draft_assistant.expert_data import ExpertBoard, ExpertValue
from sleeper_draft_assistant.roster_value import evaluate_roster
from sleeper_draft_assistant.strategy import WeeklyPlayer


def test_rank_assignment_and_forecast_independence():
    players = {pid: WeeklyPlayer(pid, pid, pos, "CHI", points, roster_value=points) for pid, pos, points in
               (("rb1", "RB", 1), ("rb2", "RB", 2), ("wr", "WR", 3))}
    values = {pid: ExpertValue(pid, p.position, rank, rank, rank, ("e1", "e2", "e3"), rank, credit)
              for pid, p, rank, credit in (("rb1", players["rb1"], 2, 2), ("rb2", players["rb2"], 3, 3), ("wr", players["wr"], 1, 4))}
    board = ExpertBoard(values, ("e1", "e2", "e3"), 3)
    before = evaluate_roster(set(players), players, ("RB", "FLEX"), board)
    changed = {pid: replace(p, points=9999, roster_value=9999, locked=True, bye=True) for pid, p in players.items()}
    assert evaluate_roster(set(changed), changed, ("RB", "FLEX"), board) == before
