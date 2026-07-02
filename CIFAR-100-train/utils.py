import sys
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Filter subdatasets
def load_cifar100_subset(train_dataset:torchvision.datasets.CIFAR100,
                         validation_dataset:torchvision.datasets.CIFAR100, 
                         target_classes:list):
    all_classes = train_dataset.classes
    try:
        # Get the original integer indices for the target class names.
        target_indices = [all_classes.index(cls) for cls in target_classes]
    # Handle the case where a specified class name is not in the dataset.
    except ValueError as e:
        print(f"Error: One of the target classes not found in CIFAR-100. {e}")
        sys.exit(1)
    
    # Create a mapping from the original class indices to new, contiguous indices (0, 1, 2, ...).
    label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}

    train_dataset = filter_dataset(train_dataset, target_classes, target_indices, label_map)
    validation_dataset = filter_dataset(validation_dataset, target_classes, target_indices, label_map)
    return train_dataset, validation_dataset

def filter_dataset(dataset, target_classes, target_indices, label_map):
    targets_np = np.array(dataset.targets)
    # Create a boolean mask to identify which samples belong to the target classes.
    indices_to_keep = np.isin(targets_np, target_indices)
    
    # Filter the dataset's image data using the boolean mask.
    dataset.data = dataset.data[indices_to_keep]
    
    # Get the original labels of the samples that are being kept.
    original_targets_to_keep = targets_np[indices_to_keep]
    # Remap the original labels to the new contiguous labels.
    dataset.targets = [label_map[target] for target in original_targets_to_keep]
    
    # Update the dataset's class list to only include the target classes.
    dataset.classes = target_classes
    return dataset

# Visualise random from datasets
def visualise_random(cifar100_datasets:torchvision.datasets.CIFAR100):
    # note: copy the dataset in order not to change the original training set.
    #       Only convert to tensor, don't do normalize.
    show_dataset = copy.copy(cifar100_datasets)
    show_dataset.transform = transforms.ToTensor()
    fig, axes = plt.subplots(3,4, figsize=(4,4))
    indecies = torch.randint(1,100, (1,12)).squeeze().numpy()
    for i, idx in enumerate(indecies):
        # print(f"i:{i//4} j:{i%4} idx:{idx}")
        img, label = show_dataset[idx]
        m,n = i//4, i%4
        axes[m,n].imshow(img.permute(1, 2, 0))
        axes[m,n].set_title(f'{show_dataset.classes[label]}', fontsize=6)
        axes[m,n].axis('off')
        
    plt.tight_layout()  # 自动调整子图间距
    plt.show()


def plot_training_results(train_losses, val_losses, val_accuracies):
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # --- Configure the first subplot for training and validation loss ---
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color="#185df3", linewidth=2.5, marker='o', markersize=3, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color="#2cd69d", linewidth=2.5, marker='o', markersize=3, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    ax2 = axes[1]
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color="#7821db", linewidth=2.5, marker='o', markersize=3, label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=0, right=num_epochs)

    plt.tight_layout()
    # Display the plots
    plt.show()