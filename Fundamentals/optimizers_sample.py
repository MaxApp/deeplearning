"""Common PyTorch usage patterns for four optimizers"""

import torch
from torch import nn


def sgd_with_momentum(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=0.01,
        momentum=0.9
    )


def rmsprop(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.RMSprop(
        model.parameters(),
        lr=0.001,
        alpha=0.99
    )


def adam(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )


def adamw(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.01
    )


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    model.train()
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()


# Example: create a model, loss function, and dataloader, then choose one:
# model = MyModel()
# criterion = nn.CrossEntropyLoss()
# optimizer = adamw(model)  # Replace with sgd_with_momentum, rmsprop, or adam.
# for epoch in range(num_epochs):
#     train_one_epoch(model, train_dataloader, criterion, optimizer)