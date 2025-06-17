# 🧪 Example: Safe CartPole

This example demonstrates how to train a **constrained DQN agent** on the classic `CartPole-v1` task with constraint-aware wrappers, SHAP-based explainability, and safe policy evaluation.

---

## 🧠 Objective

Balance the pole while **minimizing unsafe actions**, defined by custom constraints (e.g., excessive cart velocity).

---

## 📦 Setup

Make sure you’ve followed the [Installation Guide](../usage/installation.md).

---

## 🔧 1. Import Required Modules

```python
import gymnasium as gym
from agents.constrained_dqn import ConstrainedDQNAgent
from envs.wrappers import ConstraintWrapper
from evaluations.metrics import evaluate_agent
from explainability.shap_explainer import SHAPExplainer
```

---

## 🏗️ 2. Create a Safe Environment

```python
env = gym.make("CartPole-v1", render_mode=None)
safe_env = ConstraintWrapper(env, constraint_threshold=0.5)

```

--- 

## 🤖 3. Train a Constrained Agent

```python
agent = ConstrainedDQNAgent(env=safe_env, budget=1.0)
agent.train(episodes=10)


```

--- 

## 📊 4. Evaluate the Agent

```python
metrics = evaluate_agent(agent, safe_env, episodes=5)

print("=== Evaluation Metrics ===")
print("Safe episode rate:", metrics["safe_rate"])
print("Average violations:", metrics["avg_violations"])
print("Total regret:", metrics["regret"])

```

--- 

## 🔍 5. Explain a Sample Episode with SHAP

```python
explainer = SHAPExplainer(agent, safe_env)
explainer.explain_episode(max_steps=50)
```

--- 

## ✅ Output Snapshot

```bash
Episode 3: reward=1.0, penalty=0.0, violations=0.0
SHAP: ['0.04', '0.01']
Saliency: ['0.19', '0.05']

```

--- 
