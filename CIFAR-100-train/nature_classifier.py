import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import utils

# a limited subclasses instead of a full one
target_classes = [
    # Flowers
    'orchid', 'poppy', 'sunflower',
    # Mammals
    'fox', 'raccoon', 'skunk',
    # Insects
    'butterfly', 'caterpillar', 'cockroach'
]

# Model
class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        
        # layer1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # layer2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # layer3
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # flatten
        self.flatten = nn.Flatten()

        # FC layer
        # Input 32x32, after 3 pooling layers: 4x4
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.relu4 = nn.ReLU()
        # dropout
        self.dropout = nn.Dropout(0.5)

        # FC layer
        self.fc2 = nn.Linear(512, len(target_classes))

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout(x) # dropout
        x = self.fc2(x)

        return x

# training the model with module settings
def train_model(model, train_loader, loss_func, optimizer):
    # Set model to train mode
    model.train()
    # Initialize running loss
    training_loss = 0.0
    # Iterate over batches of data in the training loader
    for images, labels in train_loader:
        # Clear the gradients of all optimized variables
        optimizer.zero_grad()
        # Perform a forward pass to get model outputs
        outputs = model(images)
        # Calculate the loss
        loss = loss_func(outputs, labels)
        # Perform a backward pass to compute gradients
        loss.backward()
        # Update the model parameters
        optimizer.step()
        
        # Accumulate the training loss for the batch
        training_loss += loss.item() * images.size(0)
        
    # Calculate the average training loss for the epoch
    epoch_loss = training_loss / len(train_loader.dataset)
    # Return the epoch metrics
    return epoch_loss

def eval_model(model, val_loader, loss_func):
    # Set the model to evaluation mode
    model.eval()
    val_losses = 0.0
    correct = 0
    # Disable gradient descent
    with torch.no_grad():
        for val_images, labels in val_loader:
            outputs = model(val_images)
            batch_avg_loss = loss_func(outputs, labels)
            # accumulate total loss per batch
            val_losses += batch_avg_loss.item() * labels.size(0)
            predicted = torch.argmax(outputs, 1)
            correct += (predicted == labels).sum().item()

        # Calculate the average validation loss
        epoch_val_loss = val_losses / len(val_loader.dataset)
        # Calculate the accuracy
        epoch_accuracy = 100.0 * correct / len(val_loader.dataset)
        return epoch_val_loss, epoch_accuracy

if __name__ == "__main__":
    # constant values
    # mean, std data for CIFAR-100 in advance
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    # Training set transform
    train_transform = transforms.Compose([
        # Data augmentation
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(15), #  [-15,15]

        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    # Validation set transform
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar100_mean, cifar100_std)
    ])

    # load pre-downloaded dataset
    original_train_dataset = datasets.CIFAR100("E:\\CIFAR100分类数据集", train=True, download=False, transform=train_transform)
    original_val_dataset = datasets.CIFAR100("E:\\CIFAR100分类数据集", train=False, download=False, transform=val_transform)
    # filter sub dataset
    sub_train_sets, sub_val_sets = utils.load_cifar100_subset(original_train_dataset, original_val_dataset, target_classes)

    # test for dataloader if necessary
    # utils.visualise_random(sub_train_sets)
    # utils.visualise_random(sub_val_sets)

    batch_size = 64
    # data loader for the training set, with shuffling enabled
    train_loader = DataLoader(sub_train_sets, batch_size=batch_size, shuffle=True)
    # validation loader for test, with shuffling disabled
    val_loader = DataLoader(sub_val_sets, batch_size=batch_size, shuffle=False)

    # model initialize
    model = SimpleCNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    num_epochs = 20
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accuracy = []
    # Loop over epochs
    for epoch in range(num_epochs):
        # training
        train_loss = train_model(model, train_loader, loss_function, optimizer)
        # evaluating
        val_loss, accuracy = eval_model(model, val_loader, loss_function)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Accuracy: {accuracy:.2f}")
        
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_accuracy.append(accuracy)

    # print(f"train loss: {epoch_train_losses}")
    # print(f"val loss: {epoch_val_losses}")
    # print(f"val accuracy: {epoch_val_accuracy}")
    
    # show training and evaluation results in matplotlib
    utils.plot_training_results(epoch_train_losses, epoch_val_losses, epoch_val_accuracy)
