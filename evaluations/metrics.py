# evaluations/metrics.py


def compute_violation_rate(violation_log: list) -> float:
    """% of steps that violated a constraint in one episode"""
    if not violation_log:
        return 0.0
    return sum(violation_log) / len(violation_log)


def compute_safe_episode_rate(all_violation_logs: list) -> float:
    """% of episodes with 0 violations"""
    total_episodes = len(all_violation_logs)
    if total_episodes == 0:
        return 0.0
    safe_episodes = sum(1 for log in all_violation_logs if sum(log) == 0)
    return safe_episodes / total_episodes


def compute_regret(
    rewards: list, penalties: list, baseline_reward: float = 200.0
) -> float:
    """Sum regret due to constraint penalties over all episodes"""
    return sum((baseline_reward - (r + p)) for r, p in zip(rewards, penalties))
