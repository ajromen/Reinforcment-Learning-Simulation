from typing import List

import numpy as np


class Agent:
    def __init__(self, layer_widths: List[int]):
        self.layer_widths = layer_widths
        self.input_size = layer_widths[0]
        self.output_size = layer_widths[-1]

    def step(self, state: list[float]):
        raise NotImplementedError

    def reward(self, reward):
        raise NotImplementedError

    def episode_end(self):
        raise NotImplementedError
