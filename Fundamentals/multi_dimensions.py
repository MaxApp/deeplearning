"""
Multi-dimensional feature linear regression.
"""

import torch
import numpy as np


# y1 = w11 * x1 + w21 * x2 + b1
# y2 = w12 * x1 + w22 * x2 + b2
# e.g.
# y1 = 2 * x1 + 3 * x2 + 5
# y2 = 5 * x1 + 8 * x2 + 12


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        torch.manual_seed(42)  # Set a random seed for reproducibility.
        self.linear = torch.nn.Linear(2, 2)  # Input dimension is 2, output dimension is 2

    def forward(self, x):
        return self.linear(x)


def main():
    np.random.seed(100)  # For reproducibility
    x1_values = np.random.randint(-20, 21, 20)
    x2_values = np.random.randint(-20, 21, 20)
    x_input = torch.tensor(np.column_stack((x1_values, x2_values)), dtype=torch.float32)

    # Compute the corresponding target values.
    y1_values = 2 * x1_values + 3 * x2_values + 5
    y2_values = 5 * x1_values + 8 * x2_values + 12
    y_target = torch.tensor(np.column_stack((y1_values, y2_values)), dtype=torch.float32)

    model = LinearModel()
    print(f"Initial weights: {model.linear.weight.data.numpy().flatten()}, Bias: {model.linear.bias.data.numpy().flatten()}")

    loss_func = torch.nn.MSELoss()
    torch_optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    epochs = 1000

    for epoch in range(epochs):
        y_pred = model(x_input)  # Forward pass
        loss = loss_func(y_pred, y_target)  # Calculate loss
        torch_optimizer.zero_grad()
        loss.backward()
        torch_optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Weights: {model.linear.weight.data.numpy().flatten()}, Bias: {model.linear.bias.data.numpy().flatten()}")


if __name__ == "__main__":
    main()

