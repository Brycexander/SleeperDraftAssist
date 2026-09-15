from sleeper_draft_assistant.expert_data import ExpertBoard, ExpertValue
from sleeper_draft_assistant.expert_moves import suggest_expert_trades, suggest_expert_waivers
from sleeper_draft_assistant.strategy import WeeklyPlayer


def test_balanced_rank_trade_improves_both_rosters():
    players = {pid: WeeklyPlayer(pid, pid, "WR", "CHI", 10) for pid in ("a", "b")}
    board = ExpertBoard({"a": ExpertValue("a", "WR", 2, 2, 2, ("e1", "e2", "e3"), 2, 2),
                         "b": ExpertValue("b", "WR", 1, 1, 1, ("e1", "e2", "e3"), 1, 3)}, ("e1", "e2", "e3"), 3)
    rosters = [{"roster_id": 1, "players": ["a"]}, {"roster_id": 2, "players": ["b"]}]
    result = suggest_expert_trades(players, rosters, 1, ("WR",), board, capacity=1)
    assert result.diagnostics["candidates_evaluated"] == 1
    assert result.suggestions == []


def test_move_player_rows_expose_expert_disagreement():
    players = {"a": WeeklyPlayer("a", "a", "WR", "CHI", 10)}
    board = ExpertBoard(
        {"a": ExpertValue("a", "WR", 10, 5, 30, ("e1", "e2", "e3", "e4", "e5"), 3, 2, 25, "low")},
        ("e1", "e2", "e3", "e4", "e5"),
        30,
    )

    result = suggest_expert_waivers(
        players, [{"roster_id": 1, "players": []}], 1, ("WR",), board, capacity=1
    )

    assert result.suggestions[0]["add"]["rank_spread"] == 25
    assert result.suggestions[0]["add"]["agreement"] == "low"
