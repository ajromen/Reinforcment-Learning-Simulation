from typing import List

import numpy as np

rng = np.random.default_rng()

from src.agents.agent import Agent


class ReinforceAgent(Agent):
    def __init__(self, layer_widths: List[int]):
        super().__init__(layer_widths)

    def step(self, state: list[float]):
        return rng.uniform(-1.0, 1.0, size=self.output_size)

    def reward(self, reward):
        #save
        pass

    def episode_end(self):
        #backrpop
        pass