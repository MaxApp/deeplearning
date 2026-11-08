"""
Fit a noisy log10 curve and compare AdamW with SGD.

The script demonstrates standardizing the input feature and
Batch Normalization between each hidden linear layer and activation.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


NUM_OF_DATA = 100
LOSS_RECORDS = 60
EPOCHS = 1000


def create_data():
    x = torch.linspace(1, 100, NUM_OF_DATA).unsqueeze(1)
    # Add bounded noise to a logarithmic target.
    noise = (torch.rand(NUM_OF_DATA, 1) * 2 - 1) * 0.18
    y = torch.log10(x) + noise

    # Standardize the input feature; the target remains in its original scale.
    x_mean = x.mean(dim=0, keepdim=True)
    x_std = x.std(dim=0, keepdim=True, unbiased=False)
    x_normalized = (x - x_mean) / x_std
    return x, x_normalized, y


def create_model():
    return nn.Sequential(
        nn.Linear(1, 5),
        # Normalize hidden activations before applying the nonlinearity.
        nn.BatchNorm1d(5),
        nn.ReLU(),
        nn.Linear(5, 10),
        nn.BatchNorm1d(10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )


def create_plots(x, x_normalized, y):
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))

    axs[0].set_title("Original Data and AdamW Fit")
    axs[0].scatter(x, y)
    fit_line, = axs[0].plot([], [], c="r")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")

    axs[1].set_title("Standardized Input vs Target")
    axs[1].scatter(x_normalized, y)
    axs[1].set_xlabel("Standardized x")
    axs[1].set_ylabel("y")

    axs[2].set_title("Loss Curve")
    return fig, axs, fit_line


def train_model(model, x, y, optimizer, name, on_progress=None):
    loss_function = nn.MSELoss()
    recorded_epochs = []
    recorded_losses = []

    for epoch in range(EPOCHS):
        # Compute predictions and loss, then update model parameters.
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_function(y_hat, y)
        loss.backward()
        optimizer.step()

        if epoch <= LOSS_RECORDS:
            recorded_epochs.append(epoch)
            recorded_losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"{name} epoch:{epoch + 1} Loss:{loss.item()}")

        # Refresh the live fit occasionally instead of redrawing every epoch.
        if on_progress and (
            ((epoch + 1) <= 50 and (epoch + 1) % 2 == 0)
            or (epoch + 1) == EPOCHS
        ):
            on_progress(y_hat.detach())

    return recorded_epochs, recorded_losses


def main():
    torch.manual_seed(24)
    x, x_normalized, y = create_data()
    _, axs, fit_line = create_plots(x, x_normalized, y)

    adam_model = create_model()
    # AdamW applies weight decay separately from Adam's adaptive gradient update.
    adamw_optimizer = optim.AdamW(
        adam_model.parameters(), lr=0.01, weight_decay=0.01
    )

    def update_adamw_fit(y_hat):
        fit_line.set_data(x.numpy(), y_hat.numpy())
        plt.pause(0.8)

    adamw_epochs, adamw_losses = train_model(
        adam_model,
        x_normalized,
        y,
        adamw_optimizer,
        "AdamW",
        on_progress=update_adamw_fit,
    )

    sgd_model = create_model()
    sgd_optimizer = optim.SGD(sgd_model.parameters(), lr=0.02)
    sgd_epochs, sgd_losses = train_model(
        sgd_model, x_normalized, y, sgd_optimizer, "SGD"
    )

    axs[2].plot(adamw_epochs, adamw_losses, c="g", label="AdamW")
    axs[2].plot(sgd_epochs, sgd_losses, c="y", label="SGD")
    axs[2].legend()
    plt.show()


if __name__ == "__main__":
    main()
