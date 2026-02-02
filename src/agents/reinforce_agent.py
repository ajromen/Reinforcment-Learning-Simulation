from typing import List

import numpy as np
import torch
from torch import nn, FloatTensor, Tensor
import torch.nn.functional as F
import torch.optim as optim

rng = np.random.default_rng()

from src.agents.agent import Agent


class ReinforceAgent(Agent):
    def __init__(self, layer_widths: List[int], discount_factor: float = 0.99, lr: float = 1e-3):
        super().__init__(layer_widths)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = nn.ModuleList()
        for i in range(len(layer_widths) - 2):  # ne output
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))

        self.rewards = []
        self.saved_log_probs = []
        self.mean = nn.Linear(layer_widths[-2], layer_widths[-1])
        self.log_std = nn.Parameter(torch.zeros(layer_widths[-1]))
        self.discount_factor = discount_factor

        self.optimizer = optim.Adam(self.parameters(), lr)

        self.to(self.device)


    def step(self, state: list[float]):
        state_tensor = FloatTensor(state).to(self.device)

        # with torch.no_grad():  # manje mem trosi ako nema backprop
        x = self._forward(state_tensor)

        mean = self.mean(x)
        std = torch.exp(self.log_std)

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action).sum()

        self.saved_log_probs.append(log_prob)
        return action.detach().cpu().numpy()  # cpu numpy ne zna za gpu

    def _forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = F.leaky_relu(layer(x))
        return x

    def reward(self, reward):
        self.rewards.append(reward)

    def episode_end(self):
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.discount_factor * G
            returns.insert(0, G)

        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = torch.stack([
            -log_prob * G
            for log_prob, G in zip(self.saved_log_probs, returns)
        ]).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.rewards.clear()
        self.saved_log_probs.clear()
