# 🔐 SafeRL-Lite

A **lightweight, explainable, and modular** Python library for **Constrained Reinforcement Learning (Safe RL)** with real-time **SHAP & saliency-based explainability**, custom metrics, and Gym-compatible wrappers.

<p align="center">
  <img src="https://img.shields.io/github/license/satyamcser/saferl-lite?style=flat-square">
  <img src="https://img.shields.io/github/stars/satyamcser/saferl-lite?style=flat-square">
  <img src="https://img.shields.io/pypi/v/saferl-lite?style=flat-square">
  <img src="https://img.shields.io/github/actions/workflow/status/satyamcser/saferl-lite/ci.yml?branch=main&style=flat-square">
</p>

---

## 🌟 Overview

**SafeRL-Lite** empowers reinforcement learning agents to act under **safety constraints**, while remaining **interpretable** and **modular** for fast experimentation. It wraps standard Gym environments and DQN-based agents with:

- ✅ Safety constraint logic
- 🔍 Visual explainability (SHAP, saliency maps)
- 📊 Violation and reward tracking
- 🧪 Built-in testing and evaluations

---

## 🔧 Installation

> 📦 PyPI (coming soon)
```bash
pip install saferl-lite
```

## 🛠️ From source:

```bash
git clone https://github.com/satyamcser/saferl-lite.git
cd saferl-lite
pip install -e .
```

## 🚀 Quickstart
Train a constrained DQN agent with saliency-based explainability:

```bash
python train.py --env CartPole-v1 --constraint pole_angle --explain shap
```

🔹 This:

- Adds a pole-angle constraint wrapper to the Gym env

- Logs violations

- Displays SHAP or saliency explanations for agent decisions

## 🧠 Features
#### ✅ Constrained RL
- Add custom constraints via wrapper or logic class

- Violation logging and reward shaping

- Safe vs unsafe episode tracking

#### 🔍 Explainability
- SaliencyExplainer — gradient-based visual heatmaps

- SHAPExplainer — feature contribution values per decision

- Compatible with any PyTorch-based agent

#### 📊 Metrics
- Constraint violation rate

- Episode reward

- Cumulative safe reward

- Action entropy & temporal behavior stats

#### 📚 Modularity
- Swap out agents, constraints, evaluators, or explainers

- Supports Gym environments

- Configurable training pipeline

## 📜 Citation
Coming soon after arXiv/preprint release.