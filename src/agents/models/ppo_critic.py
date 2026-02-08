import torch
from torch import nn, Tensor
import torch.nn.functional as F

class PPOCritic(nn.Module):
    def __init__(self, layer_widths, lr, optim):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):  # ne output
            self.layers.append(nn.Linear(layer_widths[i], layer_widths[i + 1]))

            self.description = {"Layer Widths": layer_widths,
                                "Learning Rate": f"{lr:.0e}",
                                "Activation": "Leaky ReLU",
                                "Optimizer": optim,
                                "Number of Parameters": sum(p.numel() for p in self.parameters())}

    def forward(self, x: Tensor):
        for layer in self.layers[:-1]:
            x = F.leaky_relu(layer(x))
        return self.layers[-1](x).squeeze(-1)