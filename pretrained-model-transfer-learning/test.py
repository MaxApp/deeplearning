import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import utils

transform = transforms.Compose([
    transforms.ToTensor(),
])

# MNIST train data
train_dataset = torchvision.datasets.MNIST(root="./data",train=True,download=False,transform=None)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
utils.print_gray_image_in_console(train_dataset[3][0])
# utils.plot_gray_image(train_dataset[9][0])