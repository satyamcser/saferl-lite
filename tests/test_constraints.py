from agents.constraints import ActionBudgetConstraint


def test_action_budget_constraint():
    constraint = ActionBudgetConstraint(max_actions=3)

    penalties = []
    for _ in range(5):
        penalty = constraint.compute_penalty(state=None, action=0, reward=1.0)
        penalties.append(penalty)

    assert penalties[:3] == [0.0, 0.0, 0.0]
    assert penalties[3:] == [1.0, 1.0]

    constraint.reset()
    penalty_reset = constraint.compute_penalty(None, 0, 1.0)
    assert penalty_reset == 0.0
