
---

### ✅ `docs/constraints.md`: Constraint Functions

```markdown
# 🚧 Constraint Module

SafeRL-Lite allows defining **state-based constraints** that signal whether an agent's behavior violates safety, fairness, energy, or risk-related bounds.

---

## 📦 Base Class: `Constraint`

```python
from agents.constraints import Constraint
```
### ✅ Interface
```python
class Constraint:
    def __init__(self, threshold: float)
    def is_violated(self, state: np.ndarray) -> bool
    def penalty(self, state: np.ndarray) -> float
```

- is_violated: Returns True if constraint is breached.

- penalty: Returns a penalty score (e.g., distance from threshold).

---

## ⚙️ Examples
### CartPoleAngleConstraint
```python

CartPoleAngleConstraint(threshold=0.2)
```
Checks if pole angle exceeds ±threshold.

### CartPolePositionConstraint
```python
CartPolePositionConstraint(threshold=2.4)
```
Checks if cart position exceeds ±threshold.

### 🔧 Use in Agent
During training:

- Each constraint is checked per state.

- Penalty is subtracted from reward or used in constraint loss.
---

#### 🛠️ Future Plans
- Support constraint composition (AND/OR)

- Reward shaping via Lagrangian dual