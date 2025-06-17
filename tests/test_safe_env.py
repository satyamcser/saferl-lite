import gymnasium as gym
from envs.wrappers import SafeEnvWrapper


def test_safe_env_wrapper_step():
    env = SafeEnvWrapper(gym.make("CartPole-v1"), max_force=8.0)
    obs, _ = env.reset()
    next_obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(obs, (list, tuple)) or hasattr(obs, "shape")
    assert isinstance(env.violation_log, list)
    assert all(v in [0.0, 1.0] for v in env.violation_log)
