import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


class MyMNSITClassifier(nn.Module):
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


def train_model(model, device, dataloader, epochs):
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train() # 训练模式

    for epoch in range(epochs):
        total = 0
        hundred_batch_loss = 0
        hundred_correct = 0
        hundred_total = 0
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

            if (batch_idx + 1) % 100 == 0:
                print(f"{total}/60000")
                print(f"AvgLoss in batch period: {hundred_batch_loss/100:.3f}  Accuracy: {hundred_correct/hundred_total:.3f}")
                hundred_correct = 0
                hundred_total = 0
                hundred_batch_loss = 0


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
    model = MyMNSITClassifier().to(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081))
    ])

    PERIODS = 3 # 训练 + 验证周期
    EPOCHS = 2 # 每轮训练周期

    # MNIST train data
    train_dataset = torchvision.datasets.MNIST(
                    root="./data",train=True,download=True,transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    # MNIST test data
    test_dataset = torchvision.datasets.MNIST(
                    root="./data",train=False,download=True,transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    for i in range(PERIODS):
        train_model(model, device, train_loader, EPOCHS)
        test_model(model, device, test_loader)
            

