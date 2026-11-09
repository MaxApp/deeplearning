"""
# Simple manual gradient descent example for a linear model.
#
# Model: y = w1 * x1 + w2 * x2 + b
# Data format: [[x1, x2, y]]
# Target parameters: w1 = 2, w2 = 3, b = 5
"""

import random
import matplotlib.pyplot as plt

# [x1, x2, y]
data = [
    [0, 0, 5],
    [1, 0, 7],
    [0, 1, 8],
    [1, 1, 10],
    [2, 0, 9],
    [0, 2, 11],
    [2, 1, 12],
    [1, 2, 13],
    [3, 0, 11],
    [0, 3, 14],
    [3, 1, 14],
    [1, 3, 16],
    [4, 0, 13],
    [0, 4, 17],
    [2, 2, 15],
]

# Set a random seed for reproducibility.
random.seed(42)
# Randomly initialize w1, w2, and b.
w1 = random.uniform(0, 10)
w2 = random.uniform(0, 10)
b = random.uniform(0, 10)
print(f"Initial weights: w1={w1}, w2={w2}, b={b}")

# Learning rate.
lr = 0.001

# Forward propagation function.
def forward(x1, x2):
    return w1 * x1 + w2 * x2 + b

# Loss function.
def loss_func(y_pred, y_true):
    return (y_pred - y_true) ** 2

# Number of training epochs.
epochs = 10000
# Record the epoch number and MSE during training.
epoch_losses = []

if __name__ == "__main__":
    # Train the model.
    for epoch in range(1, epochs + 1):
        # Randomly sample 10 data points from the dataset.
        sample_data = random.sample(data, 10)
        # Uncomment below to print the sampled data for debugging.
        # print("Sampled data:")
        # for x1, x2, y_true in sample_data:
        #     print(f"x1: {x1}, x2: {x2}, y_true: {y_true}")

        total_loss = 0.0
        grad_w1 = 0.0
        grad_w2 = 0.0
        grad_b = 0.0

        for x1, x2, y_true in sample_data:
            # Forward pass.
            y_pred = forward(x1, x2)

            # Accumulate the batch loss.
            loss = loss_func(y_pred, y_true)
            total_loss += loss

            # Accumulate the batch gradients.
            grad_w1 += 2 * x1 * (y_pred - y_true)
            grad_w2 += 2 * x2 * (y_pred - y_true)
            grad_b += 2 * (y_pred - y_true)

        # Update weights and bias using a mini-batch gradient descent step.
        w1 -= lr * (grad_w1 / len(sample_data))
        w2 -= lr * (grad_w2 / len(sample_data))
        b -= lr * (grad_b / len(sample_data))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Average Loss: {total_loss / len(sample_data)}, w1: {w1}, w2: {w2}, b: {b}")

        if epoch > 2000 and epoch % 100 == 0:
            # Record the epoch and mean squared error for plotting.
            epoch_losses.append((epoch, total_loss / len(sample_data)))

    # Plot the training loss curve.
    epochs, losses = zip(*epoch_losses)
    plt.plot(epochs, losses, marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Training Loss Curve')
    plt.grid()
    plt.show()

