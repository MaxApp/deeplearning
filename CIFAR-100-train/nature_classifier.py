import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

import utils

# mean, std data from CIFAR-100 in advance
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)

# Training set transformation pipeline
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.4),
    transforms.RandomRotation(15), #  [-15,15]
    transforms.ToTensor(),
    transforms.Normalize(cifar100_mean, cifar100_std)
])

target_classes = [
    # Flowers
    'orchid', 'poppy', 'sunflower',
    # Mammals
    'fox', 'raccoon', 'skunk',
    # Insects
    'butterfly', 'caterpillar', 'cockroach'
]

# load dataset
original_dataset = datasets.CIFAR100("E:\\CIFAR100分类数据集", train=True, download=False, transform=train_transform)
sub_dataset = utils.load_cifar100_training_subset(original_dataset, target_classes)

# test for dataloader if necessary
# utils.visualise_random(sub_dataset)

# Set the number of samples to be processed in each batch
batch_size = 64

# Create a data loader for the training set, with shuffling enabled
train_loader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=True)

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
        x = self.dropout(x)
        x = self.fc2(x)

        return x


model = SimpleCNN()

# Loss function
loss_function = nn.CrossEntropyLoss()

# Optimizer for the prototype model
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train_model(model, train_loader, num_epochs, loss_func, optimizer):
    train_losses = []
    print("--- Training Started ---")
    
    # Loop over the specified number of epochs
    for epoch in range(num_epochs):
        model.train()
        # Initialize running loss for the current epoch
        running_loss = 0.0
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
            running_loss += loss.item() * images.size(0)
            
        # Calculate the average training loss for the epoch
        epoch_loss = running_loss / len(train_loader.dataset)
        # Append the epoch loss to the list of training losses
        train_losses.append(epoch_loss)
        
        # Print the metrics for the current epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")
        
    # Print a message indicating the end of the training process
    print("--- Finished Training ---")
    
    # Return the trained model and the collected metrics
    return model, train_losses

if __name__ == "__main__":
    num_epochs = 15
    train_model(model, train_loader, num_epochs, loss_function, optimizer)