import gymnasium as gym
import numpy as np
import torch
import typer
import yaml
import os
import pickle

from agents.constrained_dqn import ConstrainedDQNAgent
from agents.constraints import ActionBudgetConstraint
from envs.wrappers import SafeEnvWrapper
from evaluations.metrics import (
    compute_violation_rate,
    compute_safe_episode_rate,
    compute_regret,
)
from explainability.shap_explainer import SHAPExplainer
from explainability.saliency import SaliencyExplainer

app = typer.Typer()


@app.command()
def main(config: str = typer.Option(..., help="Path to config YAML")):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["train"]["log_dir"], exist_ok=True)

    # === Environment ===
    env = gym.make(cfg["env"]["name"])
    env = SafeEnvWrapper(env, max_force=cfg["env"]["max_force"])
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # === Agent ===
    constraint = ActionBudgetConstraint(cfg["agent"]["constraint"]["max_actions"])
    agent = ConstrainedDQNAgent(state_dim, action_dim, constraint=constraint)

    # === Explainability ===
    shap_explainer = SHAPExplainer(
        agent.q_net, input_dim=state_dim, device=agent.device
    )
    saliency_explainer = SaliencyExplainer(agent.q_net, device=agent.device)

    episodes = cfg["train"]["episodes"]
    epsilon = cfg["train"]["epsilon"]
    epsilon_decay = cfg["train"]["epsilon_decay"]
    min_epsilon = cfg["train"]["min_epsilon"]

    episode_rewards, episode_penalties, violation_logs = [], [], []
    shap_episode_log, saliency_episode_log = [], []

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        total_penalty = 0
        agent.reset_constraints()

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            reward_adj, penalty = agent.apply_constraint(state, action, reward)
            agent.store_transition(state, action, reward_adj, next_state, done)
            agent.update()

            state = next_state
            total_reward += reward
            total_penalty += penalty

        epsilon = max(min_epsilon, epsilon * epsilon_decay)

        episode_rewards.append(total_reward)
        episode_penalties.append(total_penalty)
        violation_logs.append(env.violation_log)

        # === Explain current state ===
        if cfg["train"]["save_explanations"]:
            state_for_explanation = np.array(state)
            shap_vals = shap_explainer.explain(state_for_explanation)
            saliency_tensor = torch.tensor(
                state_for_explanation, dtype=torch.float32
            ).to(agent.device)
            selected_action = agent.select_action(state_for_explanation, epsilon=0.0)
            saliency_vals = saliency_explainer.explain(
                saliency_tensor, target_action=selected_action
            )

            shap_episode_log.append(shap_vals.values[0][selected_action])
            saliency_episode_log.append([v.item() for v in saliency_vals])

            print(
                f"Episode {episode + 1}: reward={total_reward:.2f}, penalty={total_penalty:.2f}, violations={sum(env.violation_log)}"
            )
            print(
                f"  SHAP:     {[f'{v:.3f}' for v in shap_vals.values[0][selected_action]]}"
            )
            print(f"  Saliency: {[f'{v:.3f}' for v in saliency_vals]}")
            print("-" * 60)

    env.close()

    # === Save logs ===
    with open(
        os.path.join(cfg["train"]["log_dir"], "cartpole_run_logs.pkl"), "wb"
    ) as f:
        pickle.dump(
            {
                "rewards": episode_rewards,
                "penalties": episode_penalties,
                "violations": violation_logs,
            },
            f,
        )

    if cfg["train"]["save_explanations"]:
        with open(os.path.join(cfg["train"]["log_dir"], "explanations.pkl"), "wb") as f:
            pickle.dump({"shap": shap_episode_log, "saliency": saliency_episode_log}, f)

    # === Summary ===
    print("\n=== Evaluation Metrics ===")
    print(f"Safe episode rate: {compute_safe_episode_rate(violation_logs) * 100:.2f}%")
    print(
        f"Avg violation rate: {np.mean([compute_violation_rate(v) for v in violation_logs]):.2f}"
    )
    print(f"Total regret: {compute_regret(episode_rewards, episode_penalties):.2f}")


if __name__ == "__main__":
    app()
