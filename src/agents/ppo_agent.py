from typing import List

import numpy as np
import torch
from sympy.physics.quantum.density import entropy
from torch import nn, FloatTensor

from src.agents.agent import Agent
from src.agents.models.ppo_critic import PPOCritic
from src.agents.models.ppo_policy import PPOPolicy
import torch.nn.functional as F


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
                 input_file: str = None,
                 clip_epsilon: float = 0.2,
                 k_epochs: int = 10,
                 entropy_coef: float = 0.01, ):
        super().__init__(layer_widths)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = PPOPolicy(layer_widths).to(self.device)
        critic_layer = list(layer_widths[:-1]) + [1]
        self.critic = PPOCritic(critic_layer).to(self.device)

        self.max_batches = batch_size

        self.batches: List[Batch] = []
        self.curr_batch = Batch()
        self.count = 0

        # hyperparams
        self.discount_factor = discount_factor
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.input_file = input_file
        self.clip_epsilon = clip_epsilon
        self.k_epochs = k_epochs
        self.entropy_coef = entropy_coef

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

    def step(self, state: list[float]) -> np.ndarray:
        state_tensor = FloatTensor(state).to(self.device)

        mean, std = self.actor(state_tensor)

        base_dist = torch.distributions.Normal(mean, std)
        transform = torch.distributions.transforms.TanhTransform()
        dist = torch.distributions.TransformedDistribution(base_dist, [transform])

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        value = self.critic(state_tensor).detach()

        self.curr_batch.append(
            Pass(
                state=state_tensor.detach(),
                action=action.detach(),
                log_prob=log_prob.detach(),
                value=value.detach().squeeze(-1),
            )
        )

        return action.detach().cpu().numpy()

    def _log_prob(self, state, action):
        mean, std = self.actor(state)
        base_dist = torch.distributions.Normal(mean, std)
        transform = torch.distributions.transforms.TanhTransform()
        dist = torch.distributions.TransformedDistribution(base_dist, [transform])
        log_prob = dist.log_prob(action).sum(dim=-1)
        return log_prob#, base_dist.entropy().sum(dim=-1)

    def reward(self, reward):
        self.curr_batch.add_reward_last(reward)

    def _update(self):
        states = []
        values = []
        actions = []
        old_log_probs = []
        rtgs = []

        for batch in self.batches:
            for p in batch.passes:
                states.append(p.state)
                values.append(p.value)
                actions.append(p.action)
                old_log_probs.append(p.log_prob)
                rtgs.append(p.rtg)

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        rtgs = torch.tensor(rtgs, device=self.device)
        values = torch.stack(values).squeeze(-1)

        advantages = (rtgs - values).detach()
        # normalizovanje
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            # actor
            log_probs = self._log_prob(states, actions)
            entropy = -log_probs.detach()

            ratios = torch.exp(log_probs - old_log_probs)
            lcpi = ratios * advantages
            clip = torch.clip(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            Lclip = torch.min(lcpi, clip)

            loss_actor = -(Lclip + self.entropy_coef * entropy).mean()

            self.actor_optim.zero_grad()
            loss_actor.backward()
            self.actor_optim.step()

            # critic
            values_pred = self.critic(states)
            loss_critic = F.mse_loss(values_pred, rtgs)

            self.critic_optim.zero_grad()
            loss_critic.backward()
            self.critic_optim.step()

    def episode_end(self):
        self._calculate_rtg(self.curr_batch)
        self.batches.append(self.curr_batch)
        self.count += 1

        if self.count == self.max_batches:
            self.count = 0
            self._update()
            self.batches.clear()

        self.curr_batch = Batch()

    def _calculate_rtg(self, batch: Batch):
        G = 0.0
        gamma = self.discount_factor

        for p in reversed(batch.passes):
            G = p.reward + gamma * G
            p.rtg = G

    def load_from_file(self, filename):
        pass
