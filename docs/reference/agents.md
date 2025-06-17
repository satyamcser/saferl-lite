# 🤖 SafeRL Agents

SafeRL-Lite provides a set of agents that are aware of constraints and optimize not only for reward but also for **constraint satisfaction**. The main agent is `ConstrainedDQN`, which extends the classic Deep Q-Network to respect runtime constraints.

---

## 📦 Class: `ConstrainedDQN`

```python
from agents.constrained_dqn import ConstrainedDQN
```
## ✅ Constructor
```python
ConstrainedDQN(state_dim: int, action_dim: int)

```
- state_dim: Dimension of environment state (e.g., 4 for CartPole).

- action_dim: Number of actions (e.g., 2 for left/right).

---

### 🔧 Core Methods
```bash
forward(state: torch.Tensor) -> torch.Tensor
```
Returns Q-values for all actions.
```bash
select_action(state: np.ndarray, epsilon: float = 0.1) -> int
```
Chooses an action using ε-greedy policy.
```bash
optimize_step(...)
```
Performs one gradient update with respect to reward and constraints.

---

### 🧠 Training Logic
The agent:

- Minimizes temporal difference (TD) error for rewards.

- Incorporates constraint penalties during optimization.

- Trains with replay buffer and target network (standard in DQN).

---

#### Future Work:
- dd ConstrainedPPO, SafeSarsa, etc.

- Implement curriculum learning with adaptive constraints.

