import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from pathlib import Path
import utils


if __name__ == "__main__":

    current_dir = Path(__file__).parent
    # read the image
    image = Image.open(current_dir / "imgs/bird.jpg")
    # convert to tensor
    image_tensor = transforms.ToTensor()(image)

    # display
    # utils.plot_image(image_tensor)

    # conv block
    conv_block = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(in_channels=16, out_channels=9, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    )
    # apply the convolutional layer to the image
    output_conv_layer = conv_block(image_tensor)

    utils.plot_channels(output_conv_layer)
