from typing import List

import numpy as np
import torch
from torch import nn


class Agent:
    def __init__(self, layer_widths: List[int], name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        self.layer_widths = layer_widths
        self.input_size = layer_widths[0]
        self.output_size = layer_widths[-1]

    def step(self, state: list[float]) -> np.ndarray:
        raise NotImplementedError

    def reward(self, reward):
        raise NotImplementedError

    def episode_end(self):
        raise NotImplementedError

    def end_simulation(self, filepath):  # tj save to file
        raise NotImplementedError

    def load_from_file(self, filename):
        pass
