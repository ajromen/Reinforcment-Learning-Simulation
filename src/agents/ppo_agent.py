from typing import List

import numpy as np
import torch
from torch import nn

from src.agents.agent import Agent

class PPOAgent(Agent):
    def __init__(self, layer_widths: List[int]):
        super().__init__(layer_widths)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = nn.ModuleList()
        for i in range(len(layer_widths) - 2):  # ne output
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))




    def step(self, state: list[float]) -> np.ndarray:
        pass

    def reward(self, reward):
        pass

    def episode_end(self):
        pass
