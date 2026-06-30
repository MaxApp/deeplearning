import sys
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Filter subdatasets
def load_cifar100_training_subset(train_dataset:torchvision.datasets.CIFAR100, target_classes:list) -> torchvision.datasets.CIFAR100:
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

    targets_np = np.array(train_dataset.targets)
    # Create a boolean mask to identify which samples belong to the target classes.
    indices_to_keep = np.isin(targets_np, target_indices)
    
    # Filter the dataset's image data using the boolean mask.
    train_dataset.data = train_dataset.data[indices_to_keep]
    
    # Get the original labels of the samples that are being kept.
    original_targets_to_keep = targets_np[indices_to_keep]
    # Remap the original labels to the new contiguous labels.
    train_dataset.targets = [label_map[target] for target in original_targets_to_keep]
    
    # Update the dataset's class list to only include the target classes.
    train_dataset.classes = target_classes
    return train_dataset

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