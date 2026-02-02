from typing import List

import numpy as np
import torch
from torch import nn, FloatTensor, Tensor
import torch.nn.functional as F
import torch.optim as optim

from src.agents.models.reinforce_policy import ReinforcePolicy

rng = np.random.default_rng()

from src.agents.agent import Agent


class ReinforceAgent(Agent):
    def __init__(self, layer_widths: List[int], discount_factor: float = 0.99, lr: float = 1e-3,
                 input_file: str = None):
        super().__init__(layer_widths)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = ReinforcePolicy(layer_widths).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        self.discount_factor = discount_factor
        self.input_file = input_file

        self.rewards = []
        self.saved_log_probs = []


    def step(self, state: list[float]):
        state_tensor = FloatTensor(state).to(self.device)

        mean, std = self.policy(state_tensor)

        base_dist = torch.distributions.Normal(mean, std)
        transform = torch.distributions.transforms.TanhTransform()
        dist = torch.distributions.TransformedDistribution(base_dist, [transform])

        action = dist.sample()
        log_prob = dist.log_prob(action).sum()

        self.saved_log_probs.append(log_prob)
        return action.detach().cpu().numpy()  # cpu numpy ne zna za gpu

    def reward(self, reward):
        self.rewards.append(reward)

    def episode_end(self):
        if len(self.rewards)==0 or len(self.saved_log_probs)==0:
            return
        returns = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)

        returns = torch.tensor(returns, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.stack(
            [-log_prob * G for log_prob, G in zip(self.saved_log_probs, returns)]
        ).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards.clear()
        self.saved_log_probs.clear()

    def end_simulation(self):
        # save params and create markdown file
        pass

    def load_from_file(self):
        pass
