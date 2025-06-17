
---

### ✅ `docs/envs.md`: SafeRL Environments & Wrappers

```markdown
# 🌍 SafeRL Environments

SafeRL-Lite uses Gym-compatible environments but applies custom wrappers to:
- Track constraint violations
- Measure cumulative penalties
- Log safety metrics alongside reward

---

## 🧩 Wrapper: `ConstraintWrapper`

```python
from envs.wrappers import ConstraintWrapper
```

## ✅ Constructor
```python
ConstraintWrapper(env: gym.Env, constraints: List[Constraint])
```
- env: A standard Gym environment (e.g., CartPole).

- constraints: A list of Constraint objects.
--- 

## ⚙️ Behavior
- On each step, it checks constraints on the new state.

- Returns info["violations"] with list of violated constraint names.

- Optionally applies reward shaping or early termination.

### 🖼️ Example
```python
from envs.wrappers import ConstraintWrapper
from agents.constraints import CartPoleAngleConstraint
import gym

env = gym.make("CartPole-v1")
constraint = CartPoleAngleConstraint(threshold=0.2)
safe_env = ConstraintWrapper(env, constraints=[constraint])

obs = safe_env.reset()
obs, reward, done, info = safe_env.step(env.action_space.sample())
print(info["violations"])
```
---

## 📈 Logging Features
- Counts total violations per episode.

- Logs constraint metrics in info dict.

- Enables safe training with minimal code change.
--- 
### 🛠️ Planned Extensions
- Dynamic constraint activation

- Weighted violation metrics

- Integration with gymnasium