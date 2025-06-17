# 🔍 SHAP Explainer in SafeRL-Lite

SafeRL-Lite includes a SHAP (SHapley Additive exPlanations) module that quantifies how **each input feature** contributes to the agent’s Q-value predictions. SHAP is model-agnostic, interpretable, and widely used in explainable AI.

---

## 📦 Class: `SHAPExplainer`

```python
from explainability.shap_explainer import SHAPExplainer
```

## ✅ Constructor

```python
SHAPExplainer(model: nn.Module, background_data: np.ndarray, n_samples: int = 100)

```
#### Arguments:

- model: A trained PyTorch model (e.g., ConstrainedDQN).

- background_data: A 2D numpy array of shape (N, state_dim) representing typical states.

- n_samples: Number of background samples to use for SHAP kernel approximation (default: 100).

---

## 🔍 Method: explain_state

```python
explain_state(state: np.ndarray) -> Dict[int, np.ndarray]


```

#### Description:
Returns SHAP values for each action output in the Q-value vector, explaining the contribution of each input feature.

#### Arguments:

- state: A single environment observation as a numpy array (e.g., shape (4,) for CartPole).

#### Returns:
- Dict[int, np.ndarray]: A dictionary mapping each action index to a SHAP values vector (same size as state).

#### Example Usage:
```python
import gym
import numpy as np
from agents.constrained_dqn import ConstrainedDQN
from explainability.shap_explainer import SHAPExplainer

# Load environment and model
env = gym.make("CartPole-v1")
model = ConstrainedDQN(state_dim=4, action_dim=2)
model.eval()

# Sample background data
background_data = np.array([env.observation_space.sample() for _ in range(100)])

# Initialize SHAP Explainer
explainer = SHAPExplainer(model=model, background_data=background_data)

# State to explain
state = env.reset()

# Get SHAP attributions
shap_values = explainer.explain_state(state)

# Print results
for action, values in shap_values.items():
    print(f"Action {action} SHAP values: {values}")
```

#### 📊 Interpreting SHAP Output:
- SHAP values quantify how each feature pushes the model’s Q-value up or down for each action.

- Positive SHAP ⇒ Feature increases Q-value for that action.

- Negative SHAP ⇒ Feature suppresses Q-value.

- Zero SHAP ⇒ Feature has no influence.

---