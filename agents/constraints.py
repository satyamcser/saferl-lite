# agents/constraints.py

from abc import ABC, abstractmethod


class Constraint(ABC):
    """Abstract base class for constraints in SafeRL agents."""

    @abstractmethod
    def compute_penalty(self, state, action, reward) -> float:
        """Return penalty value for a given step."""
        pass

    def reset(self):
        """Optional: reset internal counters for new episode."""
        pass


class ActionBudgetConstraint(Constraint):
    def __init__(self, max_actions: int):
        self.max_actions = max_actions
        self.counter = 0

    def compute_penalty(self, state, action, reward) -> float:
        self.counter += 1
        return 1.0 if self.counter > self.max_actions else 0.0

    def reset(self):
        self.counter = 0


class EnergyPenaltyConstraint(Constraint):
    def __init__(self, energy_fn, max_energy):
        self.energy_fn = energy_fn
        self.max_energy = max_energy

    def compute_penalty(self, state, action, reward) -> float:
        energy = self.energy_fn(state, action)
        return max(0.0, energy - self.max_energy)
