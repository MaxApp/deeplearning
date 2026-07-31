import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from collections import defaultdict
import glob
import os
import random
import re
from PIL import Image
import matplotlib.pyplot as plt

import utils

class SignatureTripletDataset(Dataset):
    """
    sigature triplets dataset
    """
    def __init__(self, base_data_dir, triplets_per_user=10, transform=None):
        """
        Args:
            base_data_dir: The root directory of the signature dataset.
            triplets_per_user: The number of triplets to generate per user (virtual epoch size).
            transform : transforms to apply to images.
        """
        # self.base_data_dir = base_data_dir
        self.triplets_per_user = triplets_per_user
        self.transform = transform
        self.signature_map = utils.create_signature_map(base_data_dir)
        self.user_ids = list(self.signature_map.keys())

    def __len__(self):
        return len(self.user_ids) * self.triplets_per_user

    def __getitem__(self, index):
        person_id = random.choice(self.user_ids)
        anchor_path, positive_path = random.sample(self.signature_map[person_id]['real'], 2)
        negative_path = random.choice(self.signature_map[person_id]['fake'])
        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)

        anchor_file_name = os.path.basename(anchor_path)
        positive_file_name = os.path.basename(positive_path)
        negative_file_name = os.path.basename(negative_path)
        return (anchor_img, positive_img, negative_img), (anchor_file_name, positive_file_name, negative_file_name)

    def _load_image(self, path):
        with Image.open(path) as img:
            image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)
        return image

class SimpleEmbeddingNetwork(nn.Module):
    """
    A simple Convolutional Neural Network to generate a fixed-size embedding from an image.
    """
    def __init__(self, embedding_dim=128):
        super(SimpleEmbeddingNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=5), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Dropout(0.4),
            nn.Conv2d(32,64,kernel_size=5), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Dropout(0.4),
            nn.Conv2d(64,128,kernel_size=3), nn.ReLU(), nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128*25*25,256), nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, embedding_dim)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_network):
        super().__init__()
        self.embedding_network = embedding_network
    def forward(self, *inputs, triplet_bool=True):
        if triplet_bool:
            if len(inputs) != 3:
                raise ValueError("In triplet mode, expected 3 inputs: anchor, positive, negative.")
            anchor, positive, negative = inputs
            anchor_output = self.embedding_network(anchor)
            positive_output = self.embedding_network(positive)
            negative_output = self.embedding_network(negative)
            return anchor_output, positive_output, negative_output
        else:
            if len(inputs) != 2:
                raise ValueError("In pair mode, expected 2 inputs: before_img, after_img.")
            img1, img2 = inputs
            output1 = self.embedding_network(img1)
            output2 = self.embedding_network(img2)
            return output1, output2
    def get_embedding(self, image):
        return self.embedding_network(image)



if __name__ == "__main__":

    signature_base_dir = "E:\\PDF\\pytorch\\C3M1\\signatures_train"
    # mean = [0.861, 0.861, 0.861]
    # std = [0.274, 0.274, 0.274]
    # train_transform = transforms.Compose([
    #     transforms.RandomAffine(degrees=0, shear=10, translate=(0.1,0.1)),
    #     transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    #     transforms.Resize((224,224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])
    # val_transform = transforms.Compose([
    #     transforms.Resize((224,224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=mean, std=std)
    # ])

    # embedding_dim = 128
    # embedding_net = SimpleEmbeddingNetwork(embedding_dim=embedding_dim)
    # siamese_network = SiameseNetwork(embedding_network=embedding_net)

    # triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    # optimizer_siamese = optim.AdamW(siamese_network.parameters(), lr=1e-3)
    # scheduler = optim.lr_scheduler.StepLR(optimizer_siamese, step_size=2, gamma=0.1)

    dataset = SignatureTripletDataset(base_data_dir=signature_base_dir, triplets_per_user=3)

    for i in range(0,5):
        triplet_file, triplet_filename = dataset[0]

        fig, axses = plt.subplots(nrows=1, ncols=3)
        for i, axs in enumerate(axses):
            axs.imshow(triplet_file[i])
            axs.axis('off')
            axs.set_title(triplet_filename[i])
        plt.tight_layout()
        plt.show()
