# 🧠 Saliency Visualizer in SafeRL-Lite

SafeRL-Lite includes a built-in **Saliency Visualizer** to understand how individual state features influence the agent’s decisions. This is useful for **explainability**, **debugging**, and **trust-building** in safety-critical environments.

---

## 📦 Class: `SaliencyVisualizer`

```python
from explainability.saliency import SaliencyVisualizer
```

---

## ✅ Constructor

```python
SaliencyVisualizer(model: nn.Module)

```
#### Arguments:

- model (torch.nn.Module): The trained policy network to analyze.

---

## 🔍 Method: compute_saliency

```python
compute_saliency(state: torch.Tensor, action: int) -> np.ndarray


```
#### Description:
Computes gradients of the output (Q-value for a specific action) with respect to the input state.

#### Arguments:

- state: Input state tensor, shaped like environment observation (e.g., CartPole → shape (4,)).

- action: Integer action index to compute saliency for.

#### Returns:

- np.ndarray: 1D array of gradients (same size as state) showing sensitivity of each feature.


#### Example Usage:

```python
import torch
import numpy as np
from agents.constrained_dqn import ConstrainedDQN
from explainability.saliency import SaliencyVisualizer

# Example state from CartPole
state = torch.tensor([0.1, 0.2, 0.0, -0.3], requires_grad=True).unsqueeze(0)

# Load a trained model
model = ConstrainedDQN(state_dim=4, action_dim=2)
model.eval()

# Initialize visualizer
viz = SaliencyVisualizer(model)

# Compute saliency for action 0
saliency = viz.compute_saliency(state, action=0)
print("Saliency:", saliency)
```

#### 📊 Interpreting the Output
- Each value in the returned saliency array corresponds to a state feature's influence.

- High magnitude ⇒ stronger influence on the decision.

- Useful for detecting bias or overfitting to specific state variables.

#### Tips:
- Normalize inputs for better interpretability.

- Saliency reflects first-order gradients, not causal effects.

- Use along with SHAP for richer attributions (see shap_explainer.md).

---

#### Future Plans
- Heatmap overlays for vision-based environments

- Saliency trend charts across time