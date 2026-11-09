"""Binary classification regression."""

import torch
import matplotlib.pyplot as plt


class BinaryCategory(torch.nn.Module):
    def __init__(self):
        super(BinaryCategory, self).__init__()
        self.linear = torch.nn.Linear(2, 1)

    def forward(self, xy):
        return torch.sigmoid(self.linear(xy))


def main():
    sample_num = 60
    x0_mean_value = 4.0
    x1_mean_value = 7.0
    y0_mean_value = 5.0
    y1_mean_value = 8.0
    standard = 1.2

    m_data = torch.ones(sample_num, 1)  # Create a tensor of shape (sample_num, 1) filled with ones.
    x0 = torch.normal(x0_mean_value * m_data, std=standard)
    x1 = torch.normal(x1_mean_value * m_data, std=standard)
    y0 = torch.normal(y0_mean_value * m_data, std=standard)
    y1 = torch.normal(y1_mean_value * m_data, std=standard)

    tag0 = torch.zeros(sample_num, 1)  # Labels for category 0
    tag1 = torch.ones(sample_num, 1)  # Labels for category 1
    train_x = torch.cat((x0, x1), dim=0)
    train_y = torch.cat((y0, y1), dim=0)
    tag_label = torch.cat((tag0, tag1), dim=0)

    loss_func = torch.nn.BCELoss()  # Binary cross-entropy loss function
    model = BinaryCategory()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # Use SGD optimizer

    epochs = 1800
    plt.ion()  # Enable interactive mode
    plt.xlim(-2, 10)
    plt.ylim(-2, 20)

    for epoch in range(epochs):
        # Randomly choose 40 samples from train_x and train_y for training.
        indices = torch.randperm(train_x.size(0))[:40]
        x_batch = train_x[indices].view(-1, 1)
        y_batch = train_y[indices].view(-1, 1)
        tag_batch = tag_label[indices].view(-1, 1)
        inputs = torch.cat((x_batch, y_batch), dim=1)

        outputs = model(inputs)  # Forward pass
        loss = loss_func(outputs, tag_batch)  # Compute loss
        optimizer.zero_grad()  # Clear gradients
        loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        if (epoch + 1) % 100 == 0:
            mask = outputs > 0.5
            correct = (mask == tag_batch).sum().item()
            accuracy = correct / tag_batch.size(0)
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, accuracy: {accuracy:.2%}")

            w1, w2 = model.linear.weight[0]
            b = model.linear.bias[0]
            plot_x = torch.linspace(-2, 10, 50).view(-1, 1)
            plot_y = (-w1 * plot_x - b) / w2  # Decision boundary
            plt.clf()  # Clear the current figure
            plt.xlim(-2, 10)
            plt.ylim(-2, 20)
            plt.scatter(x0.numpy(), y0.numpy(), color='blue', label='Category 0')
            plt.scatter(x1.numpy(), y1.numpy(), color='red', label='Category 1')
            plt.plot(plot_x.detach().numpy(), plot_y.detach().numpy(), color='black', label='Decision Boundary')
            plt.title('Binary Category Decision Boundary')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.pause(1.5)

    plt.show(block=True)


if __name__ == "__main__":
    main()

