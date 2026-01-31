from typing import List

from src.agents.agent import Agent


class ReinforceAgent(Agent):
    def __init__(self, layer_widths: List[int]):
        super().__init__(layer_widths)