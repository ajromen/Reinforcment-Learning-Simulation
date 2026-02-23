from pathlib import Path
from typing import List

import torch
from torch import nn, FloatTensor, Tensor
import torch.nn.functional as F
import torch.optim as optim

from src.agents.models.reinforce_policy import ReinforcePolicy

from src.agents.agent import Agent


class Batch:
    def __init__(self):
        self.rewards: List[float] = []
        self.states: List[torch.Tensor] = []
        self.actions: List[torch.Tensor] = []
        self.rtgs: List[float] = []

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_state_action(self, state, action):
        self.states.append(state.detach())
        self.actions.append(action.detach())

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
        description = """REINFORCE is a policy gradient algorithm that
updates agent policy parameters by increasing the probability of actions that resulted in higher cumulative rewards,
directly maximizing expected returns without needing a value function."""
        super().__init__(layer_widths, "REINFORCE", "Vanilla Policy Gradient", description)

        self.actor = ReinforcePolicy(layer_widths, lr, "Adam").to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.lr = lr

        self.discount_factor = discount_factor
        self.input_file = input_file

        self.batches: List[Batch] = []
        self.max_batches = batch_size
        self.count = 0
        self.curr_batch = Batch()
        self.hyperparameters = {
            "Batch Size": batch_size,
            "Discount Factor": discount_factor,
        }

    def step(self, state: list[float]):
        state_tensor = FloatTensor(state).to(self.device)

        with torch.no_grad():
            mean, std = self.actor(state_tensor)

            base_dist = torch.distributions.Normal(mean, std)
            transform = torch.distributions.transforms.TanhTransform()
            dist = torch.distributions.TransformedDistribution(base_dist, [transform])

            action = dist.sample()

        self.curr_batch.add_state_action(state_tensor, action)
        return action.detach().cpu().numpy()  # cpu numpy ne zna za gpu

    def reward(self, reward):
        self.curr_batch.add_reward(reward)

    def _update(self):
        states = []
        actions = []
        rtgs = []

        for batch in self.batches:
            batch.calculate_rtgs(self.discount_factor)
            states.extend(batch.states)
            actions.extend(batch.actions)
            rtgs.extend(batch.rtgs)

        states = torch.stack(states).to(self.device)
        actions = torch.stack(actions).to(self.device)
        rtgs = torch.tensor(rtgs, device=self.device)
        rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-8)

        mean, std = self.actor(states)
        base_dist = torch.distributions.Normal(mean, std)
        transform = torch.distributions.transforms.TanhTransform()
        dist = torch.distributions.TransformedDistribution(base_dist, [transform])
        log_probs = dist.log_prob(actions).sum(dim=-1)

        loss = -(log_probs * rtgs).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()

        return loss.cpu().detach().numpy()

    def episode_end(self):
        self.batches.append(self.curr_batch)
        self.count += 1
        to_ret = None

        if self.count == self.max_batches:
            self.count = 0
            to_ret = self._update()
            self.batches.clear()

        self.curr_batch = Batch()
        return to_ret

    def get_num_of_parameters(self):
        return sum(p.numel() for p in self.actor.parameters())

    def end_simulation(self, filepath):
        path = Path(filepath)

        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state_dict": self.actor.state_dict(),
            "optimizer_state_dict": self.actor_optimizer.state_dict(),

            "layer_widths": self.layer_widths,
            "discount_factor": self.discount_factor,
            "learning_rate": self.lr,
            "batch_size": self.max_batches,

            "device": str(self.device),
            "activation": self.actor.activation_name,
            "num_parameters": self.get_num_of_parameters(),
        }

        torch.save(checkpoint, path)

    def load_from_file(self, filename: str):
        saved = torch.load(filename, map_location=self.device)

        self.actor.load_state_dict(saved["model_state_dict"])
        self.actor_optimizer.load_state_dict(saved["optimizer_state_dict"])

        self.layer_widths = saved["layer_widths"]
        self.discount_factor = saved["discount_factor"]
        self.lr = saved["learning_rate"]
        self.max_batches = saved["batch_size"]

        self.batches.clear()
        self.curr_batch = Batch()
        self.count = 0
