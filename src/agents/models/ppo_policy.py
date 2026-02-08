import torch
from torch import nn, Tensor
import torch.nn.functional as F


class PPOPolicy(nn.Module):
    def __init__(self, layer_widths, lr, optim):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_widths) - 2):  # ne output
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))

        self.mean = nn.Linear(layer_widths[-2], layer_widths[-1])
        self.log_std = nn.Parameter(torch.zeros(layer_widths[-1]))

        self.description = {"Layer Widths": layer_widths,
                            "Learning Rate": f"{lr:.0e}",
                            "Activation": "Tanh",
                            "Optimizer": optim,
                            "Number of Parameters": sum(p.numel() for p in self.parameters())}

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = F.tanh(layer(x))
        return self.mean(x), torch.exp(self.log_std)
