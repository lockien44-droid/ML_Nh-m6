"""PyTorch MLP with a linear regression output."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        layers = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.ReLU()])
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X).squeeze(-1)

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            return self(torch.as_tensor(X, dtype=torch.float32)).numpy()
