import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MyMNISTClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128,10)
        )

    def forward(self,x):
        x = self.flatten(x)
        x = self.layers(x)
        return x


def train_model(model, device, dataloader, epochs, optimizer):
    loss_function = nn.CrossEntropyLoss()
    model.train()  # Enable training mode.

    for epoch in range(epochs):
        total = 0
        hundred_batch_loss = 0
        hundred_correct = 0
        hundred_total = 0
        batches_in_period = 0
        print("\n\n===== Epoch:", epoch+1)

        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()

            _, predicted = output.max(1)
            total += target.size(dim=0)
            hundred_total += target.size(dim=0)
            hundred_correct += predicted.eq(target).sum().item()
            hundred_batch_loss += loss.item()
            batches_in_period += 1

            if (batch_idx + 1) % 100 == 0:
                print(f"{total}/{len(dataloader.dataset)}")
                print(f"AvgLoss in batch period: {hundred_batch_loss/batches_in_period:.3f}  Accuracy: {hundred_correct/hundred_total:.3f}")
                hundred_correct = 0
                hundred_total = 0
                hundred_batch_loss = 0
                batches_in_period = 0

        if batches_in_period:
            print(f"{total}/{len(dataloader.dataset)}")
            print(f"AvgLoss in batch period: {hundred_batch_loss/batches_in_period:.3f}  Accuracy: {hundred_correct/hundred_total:.3f}")


def test_model(model, device, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data, target in dataloader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            _, predict = output.max(dim=1)
            correct += predict.eq(target).sum().item()
            total += target.size(dim=0)
    
    print("Test Accuracy:", correct/total)


if __name__ == "__main__":
    # check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MyMNISTClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081))
    ])

    PERIODS = 3
    EPOCHS = 2

    # MNIST train data
    train_dataset = torchvision.datasets.MNIST(
                    root="./data",train=True,download=True,transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # MNIST test data
    test_dataset = torchvision.datasets.MNIST(
                    root="./data",train=False,download=True,transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    for _ in range(PERIODS):
        train_model(model, device, train_loader, EPOCHS, optimizer)
        test_model(model, device, test_loader)
            
