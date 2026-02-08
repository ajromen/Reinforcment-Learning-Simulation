from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import nn, FloatTensor, Tensor
import torch.nn.functional as F
import torch.optim as optim

from src.agents.models.reinforce_policy import ReinforcePolicy

rng = np.random.default_rng()

from src.agents.agent import Agent


class Batch:
    def __init__(self):
        self.rewards: List[float] = []
        self.log_probs: List[torch.Tensor] = []
        self.rtgs: List[float] = []

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_log_prob(self, log_prob):
        self.log_probs.append(log_prob)

    def calculate_rtgs(self, discount_factor: float):
        self.rtgs = []
        G = 0.0
        for r in reversed(self.rewards):
            G = r + discount_factor * G
            self.rtgs.insert(0, G)


class ReinforceAgent(Agent):
    def __init__(self, layer_widths: List[int],
                 batch_size: int = 12,
                 discount_factor: float = 0.99,
                 lr: float = 1e-3,
                 input_file: str = None):
        super().__init__(layer_widths,"REINFORCE")

        self.policy = ReinforcePolicy(layer_widths).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.lr = lr

        self.discount_factor = discount_factor
        self.input_file = input_file

        self.batches: List[Batch] = []
        self.max_batches = batch_size
        self.count = 0
        self.curr_batch = Batch()

    def step(self, state: list[float]):
        state_tensor = FloatTensor(state).to(self.device)

        mean, std = self.policy(state_tensor)

        base_dist = torch.distributions.Normal(mean, std)
        transform = torch.distributions.transforms.TanhTransform()
        dist = torch.distributions.TransformedDistribution(base_dist, [transform])

        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)

        self.curr_batch.add_log_prob(log_prob)
        return action.detach().cpu().numpy()  # cpu numpy ne zna za gpu

    def reward(self, reward):
        self.curr_batch.add_reward(reward)

    def _update(self):
        log_probs = []
        rtgs = []

        for batch in self.batches:
            batch.calculate_rtgs(self.discount_factor)
            log_probs.extend(batch.log_probs)
            rtgs.extend(batch.rtgs)

        log_probs = torch.stack(log_probs)
        rtgs = torch.tensor(rtgs, device=self.device)
        rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-8)

        loss = -(log_probs * rtgs).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def episode_end(self):
        self.batches.append(self.curr_batch)
        self.count += 1

        if self.count == self.max_batches:
            self.count = 0
            self._update()
            self.batches.clear()

        self.curr_batch = Batch()

    def get_num_of_parameters(self):
        return sum(p.numel() for p in self.policy.parameters())

    def end_simulation(self, filepath):
        path = Path(filepath)

        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),

            "layer_widths": self.layer_widths,
            "discount_factor": self.discount_factor,
            "learning_rate": self.lr,
            "batch_size": self.max_batches,

            "device": str(self.device),
            "activation": self.policy.activation_name,
            "num_parameters": self.get_num_of_parameters(),
        }

        torch.save(checkpoint, path)

    def load_from_file(self, filename: str):
        saved = torch.load(filename, map_location=self.device)

        self.policy.load_state_dict(saved["model_state_dict"])
        self.optimizer.load_state_dict(saved["optimizer_state_dict"])

        self.layer_widths = saved["layer_widths"]
        self.discount_factor = saved["discount_factor"]
        self.lr = saved["learning_rate"]
        self.max_batches = saved["batch_size"]

        self.batches.clear()
        self.curr_batch = Batch()
        self.count = 0
