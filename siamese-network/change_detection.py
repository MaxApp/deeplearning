from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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