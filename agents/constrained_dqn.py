# agents/constrained_dqn.py

import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from agents.constraints import Constraint


class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ConstrainedDQNAgent:
    def __init__(self, state_dim, action_dim, constraint: Constraint = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-3)
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.batch_size = 64

        self.constraint = constraint
        self.action_dim = action_dim

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state)
        return q_values.argmax().item()

    def store_transition(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        s, a, r, s_next, done = zip(*batch)

        s = torch.tensor(s, dtype=torch.float32).to(self.device)
        a = torch.tensor(a).to(self.device)
        r = torch.tensor(r, dtype=torch.float32).to(self.device)
        s_next = torch.tensor(s_next, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        q_values = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze()
        with torch.no_grad():
            target_q = r + self.gamma * self.target_net(s_next).max(1)[0] * (1 - done)

        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def apply_constraint(self, state, action, reward):
        if self.constraint:
            penalty = self.constraint.compute_penalty(state, action, reward)
            return reward - penalty, penalty
        return reward, 0.0

    def reset_constraints(self):
        if self.constraint:
            self.constraint.reset()
