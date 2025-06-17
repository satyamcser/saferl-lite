# 🔐 SafeRL-Lite

A **lightweight, explainable, and modular** Python library for **Constrained Reinforcement Learning (Safe RL)** with real-time **SHAP & saliency-based explainability**, custom metrics, and Gym-compatible wrappers.

By: 
- Satyam Mishra, Vision Mentors Ltd., Hanoi, Vietnam
- Shivam Mishra, Phung Thao Vi, Vietnam National University, Hanoi, Vietnam
- Dr. Vishwanath Bijalwan, SR University, Warangal, India
- Dr. Vijay Bhaskar Semwal, MANIT, Bhopal, India
- University of West London, London, UK

<p align="center">
  <a href="https://github.com/satyamcser/saferl-lite/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/satyamcser/saferl-lite?style=flat-square" alt="License">
  </a>
  <a href="https://github.com/satyamcser/saferl-lite/stargazers">
    <img src="https://img.shields.io/github/stars/satyamcser/saferl-lite?style=flat-square" alt="Stars">
  </a>
  <a href="https://pypi.org/project/saferl-lite/">
    <img src="https://img.shields.io/pypi/v/saferl-lite?style=flat-square" alt="PyPI version">
  </a>
  <a href="https://github.com/satyamcser/saferl-lite/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/satyamcser/saferl-lite/ci.yml?branch=main&style=flat-square" alt="Build Status">
  </a>
</p>


---

## 🌟 Overview

**SafeRL-Lite** empowers reinforcement learning agents to act under **safety constraints**, while remaining **interpretable** and **modular** for fast experimentation. It wraps standard Gym environments and DQN-based agents with:

- ✅ Safety constraint logic
- 🔍 Visual explainability (SHAP, saliency maps)
- 📊 Violation and reward tracking
- 🧪 Built-in testing and evaluations

---

## ✅ Problem We Solved

Modern Reinforcement Learning (RL) agents are powerful but unsafe and opaque:

- 🚫 They frequently violate safety constraints during learning or deployment (e.g., fall off a cliff in navigation tasks).

- 😕 Their decision-making is a black box: humans can’t understand why a certain action was chosen.

- 🔍 Standard RL libraries lack native support for:

  - Enforcing hard constraints during training.

  - Explaining decisions using methods like SHAP or saliency maps.

## ✅ Our Solution
SafeRL-Lite is a lightweight Python library that:

1. 📏 Adds a SafetyWrapper around any Gym environment to enforce safety constraints (e.g., bounding actions, limiting states).

2. 🧠 Integrates explainability methods:

  - SHAPExplainer (model-agnostic local explanations).

  - SaliencyExplainer (gradient-based sensitivity maps).

3. 🔧 Wraps Constrained DQNs with ease, enabling safety-compliant Q-learning.

4. 📊 Offers built-in metrics like violation count and safe episode tracking.

## ✅ Novelty
While Safe RL and Explainable RL are separately studied, no prior lightweight library:

- Combines hard safety constraints with post-hoc interpretability.

- Is designed to be minimal, pluggable, and easily installable (pip install saferl-lite) for education, experimentation, or safe deployment.

- Enables real-time SHAP or saliency visualization for Gym-based agents out-of-the-box.
``` bash
SafeRL-Lite is the first minimal library to unify constraint satisfaction and explainability in reinforcement learning — without heavy dependencies or overhead.
```

## ✅ Our Contribution
1. 🔐 Constraint Wrapper API: Drop-in Gym wrapper for defining and enforcing logical constraints on observations, actions, and reward signals.

2. 🧠 Explainability Modules: Plug-and-play SHAP and saliency explainer classes for deep Q-networks.

3. 📦 PyPI-Ready Toolkit: Easily installed, documented, and CI/CD tested; built for research and reproducibility.

4. 📈 Metrics for Constraint Violation: Tracks unsafe episodes, per-step violations, and integrates cleanly with WandB or TensorBoard.

## ✅ Technical Explanation
- We define a custom SafeEnvWrapper(gym.Env) that:

  - Intercepts actions.

  - Applies logical rules or thresholding.

  - Optionally overrides rewards or terminations if constraints are violated.

- A ConstrainedDQNAgent uses:

  - Safety-wrapped Gym envs.

  - Standard Q-learning with optional penalty_on_violation flag.

- Post-training, the SHAPExplainer and SaliencyExplainer:

  - Generate local attributions using input perturbations or gradient norms.

  - Can visualize per-state or per-action explanations.



## ✅ Satyam's Explanation
```bash
Imagine you're teaching a robot to walk — but there’s lava on the floor!
You don’t just want it to learn fast, you want it to stay safe and explain why it stepped left, not right.
```
SafeRL-Lite is like a safety helmet and voicebox for robots:

- The helmet makes sure they don’t do dangerous stuff.

- The voicebox lets them say why they made that move.

## 🔧 Installation

> 📦 PyPI 
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