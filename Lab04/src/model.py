"""PyTorch MLP for house price regression."""
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()

        # Define hidden layers
        self.hidden_layers = nn.ModuleList()

        previous = input_dim

        for width in hidden:
            self.hidden_layers.append(
                nn.Linear(previous, width)
            )
            previous = width

        # Define output layer
        self.output_layer = nn.Linear(previous, 1)

    def forward(self, X):
        # Forward pass through hidden layers
        for layer in self.hidden_layers:
            X = torch.relu(layer(X))

        # Linear regression output
        output = self.output_layer(X)

        return output.squeeze(-1)