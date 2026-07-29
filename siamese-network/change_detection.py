from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image

class ChangeDetectionDataset(Dataset):
    """
    A PyTorch Dataset for loading 'Before' and 'After' image pairs
    for a change detection task.

    This class scans a directory where subdirectories are named
    'Positive', 'Negative', and 'No_Change', each containing 'Before' and
    'After' subfolders with corresponding image pairs.
    """
    def __init__(self, base_dir, transform=None):
        """
        Initializes the dataset by scanning the data directory and organizing file paths.
        
        Args:
            base_dir (str): Path to the root directory which contains the
                            'Positive', 'Negative', and 'No_Change' folders.
            transform (callable, optional): PyTorch transforms to be applied to each image.
        """
        self.base_dir = base_dir
        self.transform = transform
        
        # Define a mapping from class names to integer labels for convenience.
        self.class_to_label = {'Positive': 0, 'Negative': 1, 'No_Change': 2}
        
        # Build the complete list of all available image pairs from the source directory.
        self.image_pairs = self._create_image_pairs()
        
        # Raise an error if the dataset directory is empty or improperly structured.
        if not self.image_pairs:
            raise RuntimeError(f"No valid image pairs found in {base_dir}. Check directory structure.")

    def _create_image_pairs(self):
        """Scans the directory to build a list of (before_path, after_path, label) tuples (one-time setup)."""
        image_pairs = []
        # Iterate through each change category ('Positive', 'Negative', 'No_Change').
        for class_name, label in self.class_to_label.items():
            class_dir = os.path.join(self.base_dir, class_name)
            before_dir = os.path.join(class_dir, 'Before')
            after_dir = os.path.join(class_dir, 'After')
            
            # Skip this category if its 'Before' directory does not exist.
            if not os.path.isdir(before_dir):
                continue

            # Iterate through all files in the 'Before' directory.
            for filename in os.listdir(before_dir):
                if filename.lower().endswith(('.png', '.jpg')):
                    # Construct the full paths for the 'Before' and corresponding 'After' images.
                    before_path = os.path.join(before_dir, filename)
                    after_path = os.path.join(after_dir, filename)
                    
                    # Add the pair to the list only if the corresponding 'After' image exists.
                    if os.path.exists(after_path):
                        image_pairs.append((before_path, after_path, label))
        return image_pairs

    def __len__(self):
        """Returns the total number of image pairs in the dataset."""
        return len(self.image_pairs)

    def __getitem__(self, idx):
        """
        Generates and returns one pair of images and its corresponding label.
        
        Args:
            idx (int): The index of the image pair to retrieve from the dataset.
            
        Returns:
            tuple: A tuple containing (before_img, after_img, label).
        """
        # Retrieve the file paths and label for the requested index.
        before_path, after_path, label = self.image_pairs[idx]
        
        # Load the 'before' and 'after' images from their respective paths.
        before_img = self._load_image(before_path)
        after_img = self._load_image(after_path)
            
        return before_img, after_img, label
        
    def _load_image(self, path):
        """
        Helper function to robustly load a single image from a given path.

        Args:
            path (str): The file path of the image to load.

        Returns:
            The loaded and transformed image, typically a torch.Tensor.
        """
        # Use a context manager to ensure the file is properly closed after loading.
        with Image.open(path) as img:
            # Ensure the image is in RGB format, as many networks expect 3 channels.
            image = img.convert("RGB")
            # Apply any specified transformations (e.g., resizing, tensor conversion).
            if self.transform:
                image = self.transform(image)
        return image


class WeightedContrastiveLoss(nn.Module):
    """
    A contrastive loss function that incorporates class weights to handle imbalance.
    
    It adapts a multi-class problem into a binary similarity problem where
    'No_Change' is 'similar' and 'Positive'/'Negative' are 'dissimilar'.
    """
    def __init__(self, device, margin=1.0, class_weights=None):
        """
        Initializes the weighted contrastive loss function.
        
        Args:
            device (torch.device): The device to move class weights to.
            margin (float): The margin for dissimilar pairs.
            class_weights (torch.Tensor, optional): A tensor of weights for each class.
                                                      Shape: (num_classes,).
        """
        super().__init__()
        self.margin = margin
        self.device = device
        
        # Move weights to the correct device once during initialization for efficiency.
        if class_weights is not None:
            self.class_weights = class_weights.to(self.device)
        else:
            self.class_weights = None

    def forward(self, output1, output2, label):
        """
        Computes the weighted contrastive loss for a batch of embeddings.

        Args:
            output1 (torch.Tensor): Embeddings for the first set of images.
            output2 (torch.Tensor): Embeddings for the second set of images.
            label (torch.Tensor): The multi-class labels (0, 1, or 2) from the dataset.
        """
        # Calculate the pairwise Euclidean distance between the output embeddings.
        distances = F.pairwise_distance(output1, output2)
        
        # Convert multi-class labels (0, 1, 2) to binary similarity labels (1, 1, 0).
        # A label of 2 ('No_Change') is considered similar (0), others are dissimilar (1).
        binary_label = (label != 2).float()

        # Calculate the contrastive loss for each sample in the batch.
        loss_per_sample = (
            # Loss for similar pairs aims to minimize the distance.
            (1 - binary_label) * distances.pow(2) +
            # Loss for dissimilar pairs aims to make the distance larger than the margin.
            binary_label * torch.clamp(self.margin - distances, min=0).pow(2)
        )

        # Apply class-specific weights to the loss if they are provided.
        if self.class_weights is not None:
            # Gather the correct weight for each sample using its original multi-class label.
            weights = self.class_weights[label.long()]
            
            # Multiply each sample's loss by its corresponding class weight.
            loss_per_sample = loss_per_sample * weights
            
        # Return the mean of the (potentially weighted) losses for the batch.
        return loss_per_sample.mean()


if __name__ == "__main__":

    # ImageNet normalization statistics
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Transformations for the training set (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Transformations for validation set (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Initialize the custom weighted contrastive loss function with the calculated class weights.
    # contrastive_loss = WeightedContrastiveLoss(margin=2.0, class_weights=class_weights, device=device)

    # Initialize the AdamW optimizer for the new EfficientNet-based model
    optimizer_change = optim.AdamW(siamese_efficientnet.parameters(), lr=1e-3)

    # Initialize the new, more flexible scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_change,
        mode='min',      # Reduce LR when the validation loss stops decreasing
        factor=0.2,      # New LR = LR * factor
        patience=2,      # Wait 2 epochs with no improvement before reducing LR
    )