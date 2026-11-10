import os
from pathlib import Path

import scipy.io
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.transforms import transforms


class FlowersDataset(Dataset):
    def __init__(self, root_path, transform):
        self.transform = transform
        self.imgs_root = Path(root_path) / "102flowers" / "jpg"
        labels_mat = scipy.io.loadmat(os.path.join(root_path, "imagelabels.mat"))
        # Oxford Flowers102 labels are 1-based; PyTorch classification labels are 0-based.
        self.labels = labels_mat["labels"].squeeze().astype("int64") - 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = self.imgs_root / f"image_{idx + 1:05d}.jpg"
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_tensor = self.transform(image)
        return image_tensor, int(self.labels[idx])


if __name__ == "__main__":
    root_path = "/original_data"  # Replace with the path to your extracted dataset.

    train_transform = transforms.Compose([
        # Apply random augmentation only to training images.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # ImageNet statistics are commonly used with pretrained vision models.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    # Keep evaluation images deterministic and unaugmented.
    eval_dataset = FlowersDataset(root_path, eval_transform)
    train_dataset = FlowersDataset(root_path, train_transform)

    train_count = int(len(eval_dataset) * 0.70)
    val_count = int(len(eval_dataset) * 0.15)
    test_count = len(eval_dataset) - train_count - val_count
    
    train_eval_subset, val_set, test_set = random_split(
        eval_dataset,
        [train_count, val_count, test_count],
        generator=torch.Generator().manual_seed(42),
    )
    train_set = Subset(train_dataset, train_eval_subset.indices)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    print(f"Total images: {len(eval_dataset)}")
    for images, labels in train_loader:
        print(f"Image batch shape: {images.shape}")
        print(f"Label batch shape: {labels.shape}; dtype: {labels.dtype}")
        break

    print(f"Train: total -> {len(train_set)}, batches -> {len(train_loader)}")
    print(f"Validation: total -> {len(val_set)}, batches -> {len(val_loader)}")
    print(f"Test: total -> {len(test_set)}, batches -> {len(test_loader)}")
