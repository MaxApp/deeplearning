"""
多维 feature 线性回归
"""

import torch
import numpy as np
# y1 = w11*x1 + w21*x2 + b1
# y2 = w12*x1 + w22*x2 + b2
# e.g.
# y1 = 2*x1 + 3*x2 + 5
# y2 = 5*x1 + 8*x2 + 12

np.random.seed(100)  # For reproducibility
x1_values = np.random.randint(-20, 21, 20)
x2_values = np.random.randint(-20, 21, 20)
x_input = torch.tensor(np.column_stack((x1_values, x2_values)), dtype=torch.float32)

# 计算对应y值
y1_values = 2*x1_values + 3*x2_values + 5
y2_values = 5*x1_values + 8*x2_values + 12
y_target = torch.tensor(np.column_stack((y1_values, y2_values)), dtype=torch.float32)

# x = torch.tensor([[0, 1, 0, 1, 2, 0, 2, 1, 3, 0, 3, 1, 4, 0, 2],
#                   [0, 0, 1, 1, 0, 2, 1, 2, 0, 3, 1, 3, 0, 4, 2]], dtype=torch.float32)
# y = torch.tensor([[5, 8, 7, 10, 9, 11, 12, 13, 11, 14, 14, 16, 13, 17, 15]], dtype=torch.float32)
# random.seed(42)
# Create a nerual network model
class LinearModel(torch.nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        torch.manual_seed(42)  # Set random seed for reproducibility
        self.linear = torch.nn.Linear(2, 2)  # Input dimension is 2, output dimension is 2
        print(f"Initial weights: {self.linear.weight.data.numpy().flatten()}, Bias: {self.linear.bias.data.numpy().flatten()}")

    def forward(self, x):
        return self.linear(x)
    
model = LinearModel()
# Loss function
loss_func = torch.nn.MSELoss()
# Optimizer
torch_optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
# Training epochs
epochs = 1000
# Train the model
for epoch in range(epochs):
    # Forward pass
    y_pred = model(x_input)
    # print(f"y_pred shape: {y_pred.shape}, y_target shape: {y_target.shape}  ")
    loss = loss_func(y_pred, y_target)  # Calculate loss
    # Backward pass and optimize
    torch_optimizer.zero_grad()
    loss.backward()
    torch_optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Weights: {model.linear.weight.data.numpy().flatten()}, Bias: {model.linear.bias.data.numpy().flatten()}")
        # print(f"Predicted: {y_pred.data.numpy().flatten()}")
