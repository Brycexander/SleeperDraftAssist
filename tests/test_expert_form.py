from sleeper_draft_assistant.expert_data import ExpertValue
from sleeper_draft_assistant.expert_form import analyze_expert_form
from sleeper_draft_assistant.strategy import WeeklyPlayer


def test_one_game_is_provisional_sell_high():
    player = WeeklyPlayer("p", "P", "WR", "CHI", None)
    value = ExpertValue("p", "WR", 20, 20, 20, ("e1", "e2", "e3"), 20, 1)
    form = analyze_expert_form(player, value, [{"week": 1, "points": 20, "opportunities": 8, "snap_share": .8, "team": "CHI", "position_finish": 1}])
    assert form["signal"] == "sell_high" and form["provisional"] and form["confidence"] == "very_low"
    assert form["workload_change"] is None
