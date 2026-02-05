from typing import List

import numpy as np
import torch
from torch import nn, FloatTensor

from src.agents.agent import Agent
from src.agents.models.ppo_critic import PPOCritic
from src.agents.models.ppo_policy import PPOPolicy


# jedan forward pass
class Pass:
    def __init__(self, state, action, log_prob, value=0, reward=0, rtg=0):
        self.state = state
        self.action = action
        self.log_prob = log_prob
        self.value = value
        self.reward = reward
        self.rtg = rtg


class Batch:
    def __init__(self):
        self.passes: list[Pass] = []

    def append(self, p: Pass):
        self.passes.append(p)

    def add_reward_last(self, reward):
        self.passes[-1].reward = reward


class PPOAgent(Agent):
    def __init__(self,
                 layer_widths: List[int],
                 batch_size: int = 12,
                 discount_factor: float = 0.99,
                 lr_actor: float = 3e-4,
                 lr_critic: float = 1e-4,
                 input_file: str = None):
        super().__init__(layer_widths)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = PPOPolicy(layer_widths).to(self.device)
        critic_layer = list(layer_widths[:-1]) + [1]
        self.critic = PPOCritic(critic_layer).to(self.device)

        self.max_batches = batch_size

        self.batches: List[Batch] = []
        self.curr_batch = Batch()
        self.count = 0

    def step(self, state: list[float]) -> np.ndarray:
        state_tensor = FloatTensor(state).to(self.device)

        mean, std = self.actor(state_tensor)

        base_dist = torch.distributions.Normal(mean, std)
        transform = torch.distributions.transforms.TanhTransform()
        dist = torch.distributions.TransformedDistribution(base_dist, [transform])

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        value = self.critic(state_tensor).detach()

        self.curr_batch.append(Pass(state_tensor, action, log_prob, value))

        return action.detach().cpu().numpy()

    def _log_prob(self, state, action):
        mean, std = self.actor(state)
        base_dist = torch.distributions.Normal(mean, std)
        transform = torch.distributions.transforms.TanhTransform()
        dist = torch.distributions.TransformedDistribution(base_dist, [transform])
        log_prob = dist.log_prob(action).sum(dim=-1)
        return log_prob

    def reward(self, reward):
        self.curr_batch.add_reward_last(reward)

    def _update(self):
        pass

    def episode_end(self):
        self.count += 1

        if self.count == self.max_batches - 1:
            self.count = 0
            self._update()
            self.batches.clear()
            self.curr_batch = Batch()
            return

        self.batches.append(self.curr_batch)
        self.curr_batch = Batch()
