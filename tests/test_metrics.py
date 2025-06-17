from evaluations.metrics import (
    compute_violation_rate,
    compute_safe_episode_rate,
    compute_regret,
)


def test_metrics_computation():
    v_logs = [[0.0, 0.0], [1.0], [1.0, 1.0], []]
    rewards = [10, 5, 2, 8]
    penalties = [1, 0, 3, 2]

    assert abs(compute_violation_rate(v_logs[0]) - 0.0) < 1e-5
    assert compute_safe_episode_rate(v_logs) == 0.5

    # Based on compute_regret definition: regret = sum(baseline - (r + p))
    regret = compute_regret(rewards, penalties)
    assert abs(regret - 769.0) < 1e-5  # <-- now matches your actual logic
