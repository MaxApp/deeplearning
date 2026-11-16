"""PyTorch U-Net blocks for pixel-wise image segmentation."""

import random
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, cat
from torch.utils.data import DataLoader, Dataset, random_split

CLASS_NAMES = ("Background", "Glioma", "Meningioma", "Pituitary")
NUM_CLASSES = len(CLASS_NAMES)
MASK_FOREGROUND_THRESHOLD = 128
MODEL_IN_CHANNELS = 1
MODEL_BASE_CHANNELS = 32
MODEL_DEPTH = 4
MODEL_CONFIG = {
    "in_channels": MODEL_IN_CHANNELS,
    "out_channels": NUM_CLASSES,
    "base_channels": MODEL_BASE_CHANNELS,
    "depth": MODEL_DEPTH,
}
DEFAULT_CHECKPOINT_PATH = Path("./checkpoints/brain_tumor_unet.pth")
TUMOR_COLORS = ("tab:blue", "tab:orange", "tab:green")
MASK_CMAP = ListedColormap(("black", *TUMOR_COLORS))
MASK_NORM = BoundaryNorm(np.arange(-0.5, NUM_CLASSES + 0.5), MASK_CMAP.N)
TUMOR_CMAP = ListedColormap(TUMOR_COLORS)
TUMOR_NORM = BoundaryNorm(np.arange(0.5, NUM_CLASSES + 0.5), TUMOR_CMAP.N)

class DoubleConv(nn.Module):
    """Apply two 3x3 convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
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
    """Encode features and return pooled features and a skip connect tensor"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.convolutions = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.convolutions(x)
        return self.pool(skip), skip


class UpsampleBlock(nn.Module):
    """Decode features and concatenate the skip tensor"""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        # transpose convolutional
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.convolutions = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = self.upsample(x)
        # reshape with interpolate if unmatch in size (h x w)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=False
            )
        # catenate skip connection
        x = cat((skip, x), dim=1)
        return self.convolutions(x)


class UNet(nn.Module):

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_channels: int = 32, depth: int = 4):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or base_channels <= 0:
            raise ValueError("Channel counts must be positive.")
        if depth <= 0:
            raise ValueError("depth must be positive.")

        # number of channels output from each level
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

        # the last 1x1 conv
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


class BrainTumorSegmentationDataset(Dataset):
    """Load MRI images and convert grayscale masks to class-index masks."""

    def __init__(self, image_dir: Path, mask_dir: Path, image_size: int = 256):
        self.image_dir = image_dir
        self.image_size = image_size
        image_paths = sorted(
            path
            for path in image_dir.rglob("*")
            if path.suffix.lower() == ".jpg"
        )
        masks_by_name = {
            path.relative_to(mask_dir).with_suffix("").as_posix():
            path
            for path in mask_dir.rglob("*")
            if path.suffix.lower() == ".jpg"
        }
        self.samples: list[tuple[Path, Path, int]] = []
        unmatched = []
        for image_path in image_paths:
            relative_image_path = image_path.relative_to(image_dir)
            relative_name = relative_image_path.with_suffix("").as_posix() + "_m"
            mask_path = masks_by_name.get(relative_name)
            if mask_path is None:
                unmatched.append(relative_name)
            else:
                if len(relative_image_path.parts) < 2:
                    raise ValueError(
                        f"Expected image files under class folders 0-{NUM_CLASSES - 1}; "
                        f"got {image_path}."
                    )
                try:
                    class_id = int(relative_image_path.parts[0])
                except ValueError as exc:
                    raise ValueError(
                        f"Expected a numeric class folder 0-{NUM_CLASSES - 1}; "
                        f"got {relative_image_path.parts[0]!r} for {image_path}."
                    ) from exc
                
                if not 0 <= class_id < NUM_CLASSES:
                    raise ValueError(
                        f"Class folder must be an ID from 0 to {NUM_CLASSES - 1}; "
                        f"got {class_id} for {image_path}."
                    )
                self.samples.append((image_path, mask_path, class_id))

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
        image_path, mask_path, class_id = self.samples[index]
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

        # turn mask to 0 or class_id according to threshold
        mask_array = np.where(mask_array >= MASK_FOREGROUND_THRESHOLD, class_id, 0)

        if mask_array.min() < 0 or mask_array.max() >= NUM_CLASSES:
            raise ValueError(
                f"Mask values must be class IDs 0-{NUM_CLASSES - 1}; "
                f"found values from {mask_array.min()} to {mask_array.max()} "
                f"in {mask_path}."
            )

        image_tensor = torch.from_numpy(image_array).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask_array.astype(np.int64, copy=False))
        return image_tensor, mask_tensor


def display_random_samples(dataset: BrainTumorSegmentationDataset):
    """Display MRI images with black backgrounds and three tumor colors."""
    num_samples = 2
    sample_indices = random.sample(range(len(dataset)), k=min(num_samples, len(dataset)))
    figure, axes = plt.subplots(
        nrows=len(sample_indices),
        ncols=2,
        figsize=(6, 2 * len(sample_indices)),
        squeeze=False,
        constrained_layout=True
    )

    for row, index in enumerate(sample_indices):
        image, mask = dataset[index]
        axes[row, 0].imshow(image.squeeze(0).numpy(), cmap="gray")
        axes[row, 0].set_title(f"Image {index}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(
            mask.numpy(),
            cmap=MASK_CMAP,
            norm=MASK_NORM,
            interpolation="nearest",
        )
        axes[row, 1].set_title(f"Mask {index}")
        axes[row, 1].axis("off")

    _add_tumor_colorbar(figure, axes[:, 1].tolist())
    plt.show()
    plt.close(figure)


def _add_tumor_colorbar(figure, mask_axes) -> None:
    mappable = plt.cm.ScalarMappable(norm=TUMOR_NORM, cmap=TUMOR_CMAP)
    mappable.set_array([])
    colorbar = figure.colorbar(
        mappable, ax=mask_axes, ticks=range(1, NUM_CLASSES)
    )
    colorbar.ax.set_yticklabels(CLASS_NAMES[1:])


def predict_mask(model: UNet, image: Tensor) -> Tensor:
    """Predict per-pixel class IDs for a preprocessed single-channel image."""
    if image.ndim != 3 or image.shape[0] != MODEL_IN_CHANNELS:
        raise ValueError(
            f"Expected an image shaped (1, height, width); got {tuple(image.shape)}."
        )

    device = next(model.parameters()).device
    model.eval()
    with torch.inference_mode():
        logits = model(image.unsqueeze(0).to(device))
    return logits.argmax(dim=1)[0].cpu()


def display_random_prediction(
    dataset: BrainTumorSegmentationDataset,
    model: UNet,
) -> None:
    """Compare one random image's ground-truth and predicted segmentation."""
    index = random.randrange(len(dataset))
    image, target = dataset[index]
    prediction = predict_mask(model, image)

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(9, 3),
        squeeze=False,
        constrained_layout=True,
    )
    axes[0, 0].imshow(image.squeeze(0).numpy(), cmap="gray")
    axes[0, 0].set_title(f"Image {index}")
    axes[0, 1].imshow(target.numpy(), cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest")
    axes[0, 1].set_title("Ground truth")
    axes[0, 2].imshow(
        prediction.numpy(), cmap=MASK_CMAP, norm=MASK_NORM, interpolation="nearest"
    )
    axes[0, 2].set_title("Prediction")
    for axis in axes[0]:
        axis.axis("off")

    _add_tumor_colorbar(figure, axes[0, 1:].tolist())
    plt.show()
    plt.close(figure)


def run_epoch(model: UNet, data_loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None):
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
    dataset: BrainTumorSegmentationDataset,
    epochs: int = 20,
    batch_size: int = 8,
    learning_rate: float | None = None,
    checkpoint_path: Path | None = None,
    resume_from: Path | None = None,
) -> UNet:
    """Train for additional epochs, restoring optimizer state when resuming."""
    if epochs <= 0 or batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive.")
    if learning_rate is not None and learning_rate <= 0:
        raise ValueError("learning_rate must be positive when provided.")
    if dataset.image_size < 2**MODEL_DEPTH:
        raise ValueError(
            f"image_size must be at least {2**MODEL_DEPTH} for depth={MODEL_DEPTH}."
        )
    if len(dataset) < 2:
        raise ValueError("At least two image-mask pairs are required for training.")
    if checkpoint_path is None:
        checkpoint_path = resume_from or DEFAULT_CHECKPOINT_PATH

    # split dataset for training and validation
    validation_size = max(1, int(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size
    training_data, validation_data = random_split(dataset,
                                                  (training_size, validation_size),
                                                  generator=torch.Generator().manual_seed(42)
                                                  )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size)

    model = UNet(**MODEL_CONFIG).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate if learning_rate is not None else 1e-3
    )
    completed_epochs = 0

    if resume_from is not None:
        checkpoint = _read_checkpoint(resume_from, device)
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError(
                f"Checkpoint {resume_from} has no optimizer state and cannot resume "
                "training. Use it for inference or start a new training run."
            )
        if checkpoint["model_config"] != MODEL_CONFIG:
            raise ValueError("Checkpoint model configuration does not match this U-Net.")
        if checkpoint["image_size"] != dataset.image_size:
            raise ValueError(
                f"Checkpoint expects image_size={checkpoint['image_size']}, "
                f"but dataset uses image_size={dataset.image_size}."
            )
        if not isinstance(checkpoint.get("epoch"), int) or checkpoint["epoch"] < 0:
            raise ValueError("Checkpoint has an invalid completed epoch.")

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if learning_rate is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate
        completed_epochs = checkpoint["epoch"]

    final_epoch = completed_epochs + epochs
    for epoch in range(completed_epochs + 1, final_epoch + 1):
        training_loss = run_epoch(
            model, training_loader, criterion, device, optimizer
        )
        validation_loss = run_epoch(model, validation_loader, criterion, device)
        print(
            f"Epoch {epoch:02d}/{final_epoch} | "
            f"train loss: {training_loss:.4f} | "
            f"validation loss: {validation_loss:.4f}"
        )

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": MODEL_CONFIG,
            "class_names": list(CLASS_NAMES),
            "image_size": dataset.image_size,
            "epoch": final_epoch,
        },
        checkpoint_path,
    )
    print(f"Model parameters saved to {checkpoint_path}")
    return model


def load_model(
    checkpoint_path: Path,
    device: torch.device | None = None,
) -> tuple[UNet, int]:
    """Load a trained U-Net for inference; use train_model to resume training."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = _read_checkpoint(checkpoint_path, device)

    model = UNet(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, int(checkpoint["image_size"])


def _read_checkpoint(checkpoint_path: Path, device: torch.device) -> dict:
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=True
    )
    if checkpoint["class_names"] != list(CLASS_NAMES):
        raise ValueError(
            f"Checkpoint classes {checkpoint['class_names']} do not match "
            f"current classes {list(CLASS_NAMES)}."
        )
    return checkpoint


if __name__ == "__main__":

    root_dir = Path(r"E:\PDF\deep learning\brain_tumor_seg")
    dataset = BrainTumorSegmentationDataset(
        image_dir=root_dir / "image",
        mask_dir=root_dir / "mask",
        image_size=256
    )

    # To continue training, resume from a checkpoint containing optimizer state:
    # trained_model = train_model(
    #     dataset,
    #     epochs=5,
    #     resume_from=Path("./checkpoints/brain_tumor_unet.pth"),
    # )

    trained_model, _  = load_model(Path("./checkpoints/brain_tumor_unet.pth"))
    display_random_prediction(dataset, trained_model)
