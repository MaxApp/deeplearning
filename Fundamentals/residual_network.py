"""Fit a simple linear relationship with a small residual network."""

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """Two linear layers with a skip connection."""

    def __init__(self, features: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Linear(features, features)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.layers(x))


class LinearResidualNetwork(nn.Module):
    """Map one input feature to one output feature through a residual block."""

    def __init__(self, hidden_features: int = 8) -> None:
        super().__init__()
        self.input_layer = nn.Linear(1, hidden_features)
        self.residual_block = ResidualBlock(hidden_features)
        self.output_layer = nn.Linear(hidden_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        x = self.residual_block(x)
        return self.output_layer(x)


def main() -> None:
    torch.manual_seed(42)
    x = torch.linspace(-1, 1, 100).unsqueeze(1)
    y = 2 * x + 1

    model = LinearResidualNetwork()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(500):
        prediction = model(x)
        loss = loss_function(prediction, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1:3d}/500, loss: {loss.item():.6f}")

    with torch.no_grad():
        final_loss = loss_function(model(x), y).item()
    print(f"Final loss: {final_loss:.6f}")


if __name__ == "__main__":
    main()