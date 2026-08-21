import datetime
import logging
import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import Callback
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.classification import Accuracy, ConfusionMatrix

torch.set_float32_matmul_precision('medium')
logging.getLogger("mlflow").setLevel(logging.ERROR)

class CIFAR10DataModule(pl.LightningDataModule):
    """CIFAR10 dataset"""

    def __init__(self, data_dir='./CIFAR10_data', batch_size=64, num_workers=2):

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # transformations for training data
        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                               std=(0.2023, 0.1994, 0.2010)),
        ])
        
        # transformations for validation data
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                               std=(0.2023, 0.1994, 0.2010)),
        ])
        
        # CIFAR-10 labels
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')

    def prepare_data(self):
        """Downloads the CIFAR10 dataset if not exist"""
        if os.path.exists(self.data_dir) and os.path.isdir(self.data_dir):
            print("Loading from local")
        else:
            print("Downloading data")
            
        # download the dataset, will skip if already exists
        torchvision.datasets.CIFAR10(root=self.data_dir, train=True, download=True)
        torchvision.datasets.CIFAR10(root=self.data_dir, train=False, download=True)

    def setup(self):
        """
        train/val datasets
        """
        # training dataset
        self.train_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=True, transform=self.transform_train
        )
        
        # validation dataset
        self.val_dataset = torchvision.datasets.CIFAR10(
            root=self.data_dir, train=False, transform=self.transform_test
        )
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers
        )