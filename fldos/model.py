"""Tabular MLP shared by every client and the global model."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, n_in: int, n_out: int, hidden=(128, 64), dropout: float = 0.1):
        super().__init__()
        layers, prev = [], n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def get_vector(model: nn.Module) -> torch.Tensor:
    return nn.utils.parameters_to_vector(model.parameters()).detach().clone()


@torch.no_grad()
def set_vector(model: nn.Module, vec: torch.Tensor) -> None:
    """Copy vec into the model's parameters. (nn.utils.vector_to_parameters would make the parameters
    views of vec, so training the model would silently modify the caller's vector.)"""
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.copy_(vec[i:i + n].view_as(p))
        i += n
