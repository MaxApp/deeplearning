"""
PyTorch version of basic_manual_gradient.py.
"""

import torch


class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(2, 1)  # Input dimension: 2, output dimension: 1

    def forward(self, x):
        return self.linear(x)


def main():
    # y = w1 * x1 + w2 * x2 + b
    # y = 2 * x1 + 3 * x2 + 5
    x = torch.tensor(
        [
            [0, 1, 0, 1, 2, 0, 2, 1, 3, 0, 3, 1, 4, 0, 2],
            [0, 0, 1, 1, 0, 2, 1, 2, 0, 3, 1, 3, 0, 4, 2],
        ],
        dtype=torch.float32,
    )
    y = torch.tensor([5, 8, 7, 10, 9, 11, 12, 13, 11, 14, 14, 16, 13, 17, 15], dtype=torch.float32)

    lr = 0.001
    model = LinearModel()

    # Loss function.
    loss_func = torch.nn.MSELoss()
    # Parameter optimizer.
    torch_optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    epochs = 1000
    for epoch in range(epochs):
        y_pred = model(x)  # Forward pass
        loss = loss_func(y_pred.squeeze(), y)  # Compute loss
        torch_optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backward pass
        torch_optimizer.step()  # Update parameters
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    print(f"Final weights: w1={model.linear.weight[0][0].item()}, w2={model.linear.weight[0][1].item()}, b={model.linear.bias.item()}")


if __name__ == "__main__":
    main()
