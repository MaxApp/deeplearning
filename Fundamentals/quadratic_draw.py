"""
Quadratic function curve fitting.
Model: y = 3 * x^2 + 20
"""

import matplotlib.pyplot as plt
import torch


def plot_function(x, y, y_pred, title='Function Plot', xlabel='x', ylabel='y'):
    plt.figure(figsize=(10, 6))
    plt.scatter(x.detach().numpy(), y.detach().numpy(), color='blue', label='Data Points')
    plt.plot(x.detach().numpy(), y_pred.detach().numpy(), label='y_pred', color='red')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.pause(1.5)
    plt.close()


class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        torch.manual_seed(42)  # Set a random seed for reproducibility.
        self.linear1 = torch.nn.Linear(1, 10)
        self.linear2 = torch.nn.Linear(10, 20)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        return self.linear(x)


def main():
    # Generate data.
    x = torch.linspace(-10, 10, 50).view(-1, 1)  # Generate 50 points from -10 to 10.
    y = 3 * (x ** 2) + 20 + (5 * torch.randn(1, 50)).view(-1, 1)  # Compute corresponding y values.

    model = MyNetwork()
    loss_func = torch.nn.MSELoss()
    torch_optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 1500

    for epoch in range(epochs):
        y_pred = model(x)  # Forward pass
        loss = loss_func(y_pred, y)  # Compute loss
        torch_optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backward pass
        torch_optimizer.step()  # Update parameters
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
            plot_function(x, y, y_pred)


if __name__ == "__main__":
    main()

