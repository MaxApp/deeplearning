"""PyTorch U-Net blocks for pixel-wise image segmentation."""

from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, cat
from torch.utils.data import DataLoader, Dataset, random_split


class DoubleConv(nn.Module):
    """Apply two 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.layers(x)


class DownsampleBlock(nn.Module):
    """Extract encoder features and return pooled features plus a skip tensor."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolutions = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.convolutions(x)
        return self.pool(skip), skip


class UpsampleBlock(nn.Module):
    """Upsample decoder features, join the matching skip, and refine them."""

    def __init__(
        self, in_channels: int, skip_channels: int, out_channels: int
    ) -> None:
        super().__init__()
        # transpose convolutional
        self.up = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.convolutions = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.up(x)
        # reshape if unmatch in size
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        x = cat((skip, x), dim=1)
        return self.convolutions(x)


class UNet(nn.Module):
    """Encoder-decoder U-Net that returns per-pixel logits."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, 
                 base_channels: int = 32, depth: int = 4) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")

        encoder_channels = [base_channels * (2**level) for level in range(depth)]
        self.depth = depth
        self.down_blocks = nn.ModuleList()

        current_channels = in_channels
        for channels in encoder_channels:
            self.down_blocks.append(DownsampleBlock(current_channels, channels))
            current_channels = channels

        self.bottleneck = DoubleConv(current_channels, current_channels * 2)
        current_channels *= 2

        self.up_blocks = nn.ModuleList()
        for skip_channels in reversed(encoder_channels):
            self.up_blocks.append(
                UpsampleBlock(current_channels, skip_channels, skip_channels)
            )
            current_channels = skip_channels

        self.output = nn.Conv2d(current_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        minimum_size = 2**self.depth
        if x.shape[-2] < minimum_size or x.shape[-1] < minimum_size:
            raise ValueError(
                f"Input height and width must be at least {minimum_size} "
                f"for depth={self.depth}; got {tuple(x.shape[-2:])}."
            )

        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)

        x = self.bottleneck(x)
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)

        return self.output(x)


CLASS_NAMES = ("No Tumor", "Glioma", "Meningioma", "Pituitary")
NUM_CLASSES = len(CLASS_NAMES)
DATASET_FOLDER = "Brain Tumor Segmentation Dataset"


class BrainTumorSegmentationDataset(Dataset):
    """Load paired MRI images and class-index masks from the Kaggle layout."""

    def __init__(
        self, image_dir: Path, mask_dir: Path, image_size: int = 256
    ) -> None:
        self.image_dir = image_dir
        self.image_size = image_size
        image_paths = sorted(
            path
            for path in image_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        masks_by_name = {
            path.relative_to(mask_dir).with_suffix("").as_posix(): path
            for path in mask_dir.rglob("*")
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        }

        self.samples = []
        unmatched = []
        for image_path in image_paths:
            relative_name = (
                image_path.relative_to(image_dir).with_suffix("").as_posix()
            )
            mask_path = masks_by_name.get(relative_name)
            if mask_path is None:
                unmatched.append(relative_name)
            else:
                self.samples.append((image_path, mask_path))

        if not self.samples:
            raise ValueError(f"No paired images and masks found under {image_dir}.")
        if unmatched:
            raise ValueError(
                f"{len(unmatched)} image(s) have no mask with a matching relative "
                f"path under {mask_dir}; first unmatched image: {unmatched[0]}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        image_path, mask_path = self.samples[index]
        size = (self.image_size, self.image_size)

        with Image.open(image_path) as image:
            image = image.convert("L").resize(size, Image.Resampling.BILINEAR)
            image_array = np.asarray(image, dtype=np.float32).copy() / 255.0

        with Image.open(mask_path) as mask:
            mask = mask.resize(size, Image.Resampling.NEAREST)
            mask_array = np.asarray(mask).copy()

        if mask_array.ndim == 3:
            if not np.all(mask_array == mask_array[..., :1]):
                raise ValueError(
                    f"Expected a single-channel class-index mask, got {mask_path}."
                )
            mask_array = mask_array[..., 0]

        if mask_array.min() < 0 or mask_array.max() >= NUM_CLASSES:
            raise ValueError(
                f"Mask values must be class IDs 0-{NUM_CLASSES - 1}; "
                f"found values from {mask_array.min()} to {mask_array.max()} "
                f"in {mask_path}."
            )

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array.astype(np.int64, copy=False))
        return image_tensor, mask_tensor


def run_epoch(
    model: UNet,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    total_loss = 0.0

    with torch.set_grad_enabled(is_training):
        for images, masks in data_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            if optimizer is not None:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)

    return total_loss / len(data_loader.dataset)


def train_model(
    dataset_root: Path,
    output_path: Path,
    epochs: int = 20,
    batch_size: int = 8,
    image_size: int = 256,
    learning_rate: float = 1e-3,
) -> None:
    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")
    if image_size < 2**4:
        raise ValueError("image_size must be at least 16 for the default U-Net depth.")

    dataset_dir = dataset_root / DATASET_FOLDER
    dataset = BrainTumorSegmentationDataset(
        image_dir=dataset_dir / "image",
        mask_dir=dataset_dir / "mask",
        image_size=image_size,
    )
    if len(dataset) < 2:
        raise ValueError("At least two image-mask pairs are required for training.")

    validation_size = max(1, int(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size
    training_data, validation_data = random_split(
        dataset,
        (training_size, validation_size),
        generator=torch.Generator().manual_seed(42),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    training_loader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True, pin_memory=pin_memory
    )
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, pin_memory=pin_memory
    )

    model = UNet(in_channels=1, out_channels=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_validation_loss = float("inf")

    for epoch in range(1, epochs + 1):
        training_loss = run_epoch(
            model, training_loader, criterion, device, optimizer
        )
        validation_loss = run_epoch(model, validation_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train loss: {training_loss:.4f} | "
            f"validation loss: {validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": image_size,
                    "epoch": epoch,
                    "validation_loss": validation_loss,
                },
                output_path,
            )

    print(f"Best model saved to {output_path}")


def main() -> None:
    train_model(
        dataset_root=Path("./dataset"),
        output_path=Path("./checkpoints/brain_tumor_unet.pth"),
        epochs=20,
        batch_size=8,
        image_size=256,
        learning_rate=1e-3,
    )


if __name__ == "__main__":
    main()