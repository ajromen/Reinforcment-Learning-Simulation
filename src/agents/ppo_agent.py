from typing import List

import numpy as np

from src.agents.agent import Agent


class PPOAgent(Agent):
    def __init__(self, layer_widths: List[int]):
        super().__init__(layer_widths)

    def step(self, state: list[float]) -> np.ndarray:
        pass

    def reward(self, reward):
        pass