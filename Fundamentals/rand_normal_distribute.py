
"""
Generate uniform and normal distribution data and draw histograms.
"""

import torch
import matplotlib.pyplot as plt


def main():
    # Generate data.
    uniform_data = torch.rand(10000)  # Uniform distribution on [0, 1)
    normal_data = torch.randn(10000)  # Standard normal distribution

    # Create a figure.
    plt.figure(figsize=(12, 5))

    # Draw the uniform distribution.
    plt.subplot(1, 2, 1)
    plt.hist(uniform_data.numpy(), bins=50, color='skyblue', edgecolor='black')
    plt.title('torch.rand() - Uniform Distribution')
    plt.xlabel('Value Range [0,1)')
    plt.ylabel('Frequency')

    # Draw the normal distribution.
    plt.subplot(1, 2, 2)
    plt.hist(normal_data.numpy(), bins=50, color='salmon', edgecolor='black')
    plt.title('torch.randn() - Normal Distribution')
    plt.xlabel('Value Range (-∞,+∞)')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
