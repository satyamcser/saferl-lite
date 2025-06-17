# envs/wrappers.py

import gymnasium as gym
import numpy as np


class SafeEnvWrapper(gym.Wrapper):
    def __init__(self, env, max_force: float = None, max_energy: float = None):
        super().__init__(env)
        self.max_force = max_force
        self.max_energy = max_energy
        self.violation_log = []
        self.episode_log = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.violation_log.clear()
        self.episode_log.clear()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        violation = 0.0

        # Constraint: limit max force (CartPole)
        if self.max_force is not None:
            force = self._get_force(action)
            if abs(force) > self.max_force:
                violation = 1.0
                reward -= 1.0  # penalize

        # Note: remove _get_energy for now (MountainCar support will come later)

        self.violation_log.append(violation)
        self.episode_log.append(
            {
                "obs": obs,
                "action": action,
                "reward": reward,
                "violation": violation,
            }
        )

        return obs, reward, terminated, truncated, info

    def _get_force(self, action):
        # Assumes CartPole: force is ±10
        return 10.0 if action == 1 else -10.0

    def _get_energy(self, prev_obs, action, next_obs):
        if prev_obs is None:
            return 0.0
        # Simplified energy calculation: KE + PE
        velocity = next_obs[1]
        height = np.cos(3 * next_obs[0])  # approximates potential
        return 0.5 * velocity**2 + 9.8 * height
