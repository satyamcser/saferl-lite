# 📊 SafeRL-Lite Evaluation Metrics

SafeRL-Lite provides built-in evaluation metrics to assess both **performance** and **safety** of RL agents in constrained environments.

---

## 🎯 1. `evaluate_safety_violations`

```python
evaluate_safety_violations(violations: List[int]) -> float
```

#### Description:
Calculates the percentage of time steps where the agent violated a safety constraint.

#### Arguments:

- violations (List[int]): A binary list where 1 indicates a violation at that timestep.

Returns:

- float: Fraction of violated steps (e.g. 0.12 means 12% violation rate).

### Usage Example:

```python
violations = [0, 1, 0, 1, 1]
violation_rate = evaluate_safety_violations(violations)
# Output: 0.6
```
---

## 📈 2. evaluate_cumulative_reward

```python
evaluate_cumulative_reward(rewards: List[float]) -> float

```

#### Description:
Computes the total accumulated reward across an episode.

#### Arguments:

- rewards (List[float]): A list of numerical rewards at each timestep.

Returns:

- float: Total reward over the episode.

### Usage Example:

```python
rewards = [1.0, -0.2, 2.0, 0.5]
total_reward = evaluate_cumulative_reward(rewards)
# Output: 3.3
```


---

## ⚖️ 3. evaluate_constraint_satisfaction_rate

```python
evaluate_constraint_satisfaction_rate(satisfied: List[bool]) -> float
```

#### Description:
Measures the fraction of time steps where all constraints were satisfied.

#### Arguments:

- satisfied (List[bool]): A list where True means constraints were satisfied at that step.

Returns:

- float: Constraint satisfaction rate (e.g. 0.9 means 90% of steps were constraint-safe).

### Usage Example:

```python
satisfied = [True, False, True, True]
rate = evaluate_constraint_satisfaction_rate(satisfied)
# Output: 0.75

```



---

## 📊 4. Visualization Hooks (WIP)

Future versions will include utilities for:

- Violation trend plots

- Reward-violation scatter plots

- Safety-performance Pareto frontiers

#### 🧠 Pro Tip
These metrics are modular, you can log them live during training or compute them post hoc from rollout logs using:

```python
from evaluations.metrics import (
    evaluate_safety_violations,
    evaluate_cumulative_reward,
    evaluate_constraint_satisfaction_rate,
)

```

