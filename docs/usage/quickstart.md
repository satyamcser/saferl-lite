### 🚀 `docs/usage/quickstart.md`
```markdown
### 🚀 Quickstart

This guide helps you train and explain a constrained RL agent in a few lines of code.
```
---

### 🧠 Step 1: Import Modules

```python
from agents.constrained_dqn import ConstrainedDQNAgent
from envs.wrappers import ConstraintWrapper
import gymnasium as gym
```
### 🧪 Step 2: Wrap the Environment

```python
env = gym.make("CartPole-v1")
safe_env = ConstraintWrapper(env)

```
### 🤖 Step 3: Train a Constrained DQN Agent

```python
agent = ConstrainedDQNAgent(env=safe_env)
agent.train(episodes=10)
```

### 🔍 Step 4: Explain with SHAP or Saliency

```python
from explainability.shap_explainer import SHAPExplainer

explainer = SHAPExplainer(agent, safe_env)
explainer.explain_episode()

```

### 📈 Step 5: Evaluate Agent

```python
from evaluations.metrics import evaluate_agent

evaluate_agent(agent, safe_env, episodes=5)


```