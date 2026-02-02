import torch
from torch import nn, Tensor
import torch.nn.functional as F

class ReinforcePolicy(nn.Module):
    def __init__(self, layer_widths):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_widths) - 2):  # ne output
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))

        self.mean = nn.Linear(layer_widths[-2], layer_widths[-1])
        self.log_std = nn.Parameter(torch.zeros(layer_widths[-1]))

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = F.leaky_relu(layer(x))

        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std