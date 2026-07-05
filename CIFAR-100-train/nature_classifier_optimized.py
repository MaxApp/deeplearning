import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# more advanced tools for improving performance
import torchmetrics
from torch.optim import lr_scheduler

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

# CNN feature Block
class CNNBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_kernel_size, pool_stride):
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=padding),
            # Batch normalization
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size, pool_stride)
        )

    def forward(self, x):
        return self.conv_layer(x)
    
# Model
class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()
        
        # layer1
        self.conv1 = CNNBlock(in_channels=3, out_channels=32, kernel_size=3, padding=1, pool_kernel_size=2, pool_stride=2)
        # layer2
        self.conv2 = CNNBlock(in_channels=32, out_channels=64, kernel_size=3, padding=1, pool_kernel_size=2, pool_stride=2)
        # layer3
        self.conv3 = CNNBlock(in_channels=64, out_channels=128, kernel_size=3, padding=1, pool_kernel_size=2, pool_stride=2)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Input 32x32, after 3 pooling layers: 4x4
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(p=0.6),
            nn.Linear(512, len(target_classes))
        ) 

    def forward(self, x):
        # structure is more clear
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
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
    """
    Instead of calculating manually, 
    this time we leverage `torchmetrics` tool to evaluate:
    accuracy, precision, recall and f1-score
    """
    # Set the model to evaluation mode
    model.eval()
    val_losses = 0.0
    
    # accuracy
    accuracy_metric = torchmetrics.Accuracy(
        task="multiclass", 
        num_classes=len(target_classes), 
        average="micro"
    )

    # precision
    precision_metric = torchmetrics.Precision(
        task="multiclass",
        num_classes=len(target_classes),
        average="macro"
    )

    # recall
    recall_metric = torchmetrics.Recall(
        task="multiclass",
        num_classes=len(target_classes),
        average="macro"
    )

    # f1 score
    f1score_metric = torchmetrics.F1Score(
        task="multiclass",
        num_classes=len(target_classes),
        average="macro"
    )

    # Disable gradient descent
    with torch.no_grad():
        for val_images, labels in val_loader:
            outputs = model(val_images)
            batch_avg_loss = loss_func(outputs, labels)
            # accumulate total loss per batch
            val_losses += batch_avg_loss.item() * labels.size(0)
            predicted = torch.argmax(outputs, 1)
            
            # automate calculation
            accuracy_metric.update(predicted, labels)
            precision_metric.update(predicted, labels)
            recall_metric.update(predicted, labels)
            f1score_metric.update(predicted, labels)

        # Calculate the average validation loss
        epoch_val_loss = val_losses / len(val_loader.dataset)
        # Metric aggregation
        accuracy = accuracy_metric.compute().item()
        precision = precision_metric.compute().item()
        recall = recall_metric.compute().item()
        f1score = f1score_metric.compute().item()

        return epoch_val_loss, accuracy, precision, recall, f1score
   
if __name__ == "__main__":
    # constant values
    # mean, std data for CIFAR-100 in advance
    cifar100_mean = (0.5071, 0.4867, 0.4408)
    cifar100_std = (0.2675, 0.2565, 0.2761)

    # Training set transform
    train_transform = transforms.Compose([
        # Data augmentation
        # Randomly flip the image horizontally with a 50% probability.
        transforms.RandomHorizontalFlip(p=0.5),
        # Randomly flip the image vertically with a 50% probability.
        transforms.RandomVerticalFlip(p=0.5),
        # Rotate the image by a random angle between -15 and +15 degrees.
        transforms.RandomRotation(degrees=(-15, 15)),

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

    init_lr = 0.0005
    batch_size = 64
    # data loader for the training set, with shuffling enabled
    train_loader = DataLoader(sub_train_sets, batch_size=batch_size, shuffle=True)
    # validation loader for test, with shuffling disabled
    val_loader = DataLoader(sub_val_sets, batch_size=batch_size, shuffle=False)

    # model initialize
    model = SimpleCNN()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.0005)
    # add a scheduler to facilitate converging
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="max", factor=0.2, patience=3)

    num_epochs = 20
    epoch_train_losses = []
    epoch_val_losses = []
    epoch_val_accuracy = []

    # Loop over epochs
    for epoch in range(num_epochs):
        # training
        train_loss = train_model(model, train_loader, loss_function, optimizer)
        # evaluating
        val_loss, accuracy, precision, recall, f1score = eval_model(model, val_loader, loss_function)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f},\n \
              Accuracy: {accuracy:.2f} Precision: {precision:.2f} Recall: {recall:.2f} F1-score: {f1score:.2f}")
        
        # Update the learning rate
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(accuracy)
        else:
            scheduler.step()
        
        epoch_train_losses.append(train_loss)
        epoch_val_losses.append(val_loss)
        epoch_val_accuracy.append(accuracy)

    # print(f"train loss: {epoch_train_losses}")
    # print(f"val loss: {epoch_val_losses}")
    # print(f"val accuracy: {epoch_val_accuracy}")
    
    # show training and evaluation results in matplotlib
    utils.plot_training_results(epoch_train_losses, epoch_val_losses, epoch_val_accuracy)