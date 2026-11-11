import os
import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim

def create_emnist_dataloaders(train_dataset, test_dataset, batch_size=64):
    """
    Creates DataLoader objects for the EMNIST training and testing datasets
    """

    # Create a DataLoader for the training dataset
    train_dataloader = DataLoader(
        # Pass in the train_dataset
        dataset=train_dataset,
        # Set batch_size
        batch_size=batch_size,
        # Set shuffle
        shuffle=True
    )  

    # Create a DataLoader for the testing dataset
    test_dataloader = DataLoader(
        # Pass in the test_dataset
        dataset=test_dataset,
        # Set batch_size
        batch_size=batch_size,
        # Set shuffle
        shuffle=False
    )

    # Return the created DataLoaders
    return train_dataloader, test_dataloader

# construct models
def initialize_emnist_model(num_classes=26):
    """
    Initializes a sequential neural network model for EMNIST classification
    """

    torch.manual_seed(42)  # Set seed for reproducibility

    # Define the neural network model using nn.Sequential
    model = nn.Sequential(
        # Flatten the input images
        nn.Flatten(),
        # Middle part should not have more than 5 layers. Only Linear and ReLU layers.
        # Note: Hidden layer units should not exceed 256
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256,128),
        nn.ReLU(),
        # Final Layer: SHOULD be Linear Layer
        # Output layer with shape nn.Linear(valid inputs to the layer, num_classes outputs)
        nn.Linear(128, num_classes)
    )

    # Define the loss function (Cross-Entropy Loss for multi-class classification)
    loss_function = nn.CrossEntropyLoss()

    # Define the optimizer (Adaptive Moment Estimation)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, loss_function, optimizer

# training
def train_epoch(model, loss_function, optimizer, train_loader, device, verbose=True):
    """
    Trains the model for one epoch and calculates the average loss and accuracy
    """

    # Move the model to the specified device (CPU or GPU)
    model.to(device)
    # Set the model to training mode
    model.train()
    # Initialize running loss to 0
    running_loss = 0.0
    # Initialize the number of correct predictions to 0
    num_correct_predictions = 0
    # Initialize the total number of predictions to 0
    total_predictions = 0

    # Iterate over the batches of data from the training DataLoader
    for inputs, targets in train_loader:
        # Move the inputs and targets to the specified device
        inputs, targets = inputs.to(device), targets.to(device)

        # Shift target labels down by 1 (adjusting for EMNIST letters dataset)
        targets = targets - 1

        # Zero the gradients of the optimizer.
        optimizer.zero_grad()

        # Pass the inputs through the model to get the outputs.
        outputs = model(inputs)

        # Calculate the loss between the outputs and the targets.
        loss = loss_function(outputs, targets)

        # Perform backpropagation to calculate the gradients.
        loss.backward()

        # Update the model parameters using the optimizer.
        optimizer.step()

        # Loss 
        # Accumulate the loss value to running_loss
        loss_value = loss.item()
        # Add current loss value to the total running loss.
        running_loss += loss_value * len(targets)

        # Accuracy
        # Get the predicted indices (by taking the argmax along dimension 1 of the outputs).
        predicted_indices = outputs.argmax(dim=1)

        # Compare predicted indices to actual targets
        correct_predictions = predicted_indices.eq(targets)

        # Sum of correct predictions in the current batch.
        num_correct_in_batch = correct_predictions.sum().item()
        
        # Add correct predictions to the total correct predictions.
        num_correct_predictions += num_correct_in_batch

        # Get the batch size from the targets and add it to total predictions.
        batch_size = len(targets)
        total_predictions += batch_size
    
    # Calculate the average loss for the epoch. 
    # Weight batch losses by sample count so a smaller final batch is handled correctly.
    average_loss = running_loss / total_predictions

    # Calculate the accuracy percentage for the epoch. Multiply correct predictions by 100.
    accuracy_percentage = 100 * ( num_correct_predictions / total_predictions )

    # Conditionally print based on verbose flag
    if verbose:
        print(
            f"Epoch Loss (Avg): {average_loss:.3f} | Epoch Acc: {accuracy_percentage:.2f}%"
        )

    # Return the trained model and average loss
    return model, average_loss

# evaluate
def evaluate(model, test_loader, device, verbose=True):
    """
    Evaluates the model on the test dataset and returns the accuracy percentage
    """
    
    # Set the model to evaluate mode
    model.to(device)
    model.eval()
    # Initialize the number of correct predictions to 0
    num_correct_predictions = 0
    # Initialize the total number of predictions to 0
    total_predictions = 0 
    
    # Use torch.no_grad() context to disable gradient calculations during evaluation.
    with torch.no_grad():
        # Iterate over the batches of data from the test DataLoader
        for inputs, targets in test_loader:
            # Move the inputs and targets to the specified device
            inputs, targets = inputs.to(device), targets.to(device)

            # Shift target labels down by 1 (adjusting for EMNIST letters dataset)
            targets = targets - 1

            # Pass the inputs through the model to get the outputs.
            outputs = model(inputs)

            # Get the predicted indices (by taking the argmax along dimension 1 of the outputs).
            predicted_indices = outputs.argmax(dim=1)

            # Compare predicted indices to actual targets
            correct_predictions = predicted_indices.eq(targets)       
            
            # Sum of correct predictions in the current batch.
            num_correct_in_batch = correct_predictions.sum().item()
            
            # Add correct predictions to the total correct predictions.
            num_correct_predictions += num_correct_in_batch

            # Get the batch size from the targets and add it to total predictions.
            batch_size = len(targets)
            total_predictions += batch_size 
        
        # Calculate the accuracy percentage. Multiply correct predictions by 100.
        accuracy_percentage = (num_correct_predictions / total_predictions) * 100

    if verbose:  # Conditionally print based on verbose flag
        print((f'Test Accuracy: {accuracy_percentage:.2f}%'))

    return accuracy_percentage


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    data_path = os.path.join(os.path.dirname(__file__), "EMNIST_data")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1736,), std=(0.3317,)),
    ])

    # Let torchvision check and download any missing dataset files.
    train_dataset = datasets.EMNIST(
        root=data_path,
        split="letters",
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.EMNIST(
        root=data_path,
        split="letters",
        train=False,
        download=True,
        transform=transform,
    )

    train_loader, test_loader = create_emnist_dataloaders(
        train_dataset, test_dataset, batch_size=64
    )
    model, loss_func, optimizer = initialize_emnist_model(num_classes=26)
    num_epochs = 10

    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch + 1}")
        model, _ = train_epoch(
            model, loss_func, optimizer, train_loader, device, verbose=True
        )

    # Evaluate on the test set only after training is complete.
    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()
