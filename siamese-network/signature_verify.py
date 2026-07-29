import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from collections import defaultdict
import glob
import os
import random
from PIL import Image

class SignatureTripletDataset(Dataset):
    """
    A PyTorch Dataset for creating signature triplets for verification.
    """
    def __init__(self, base_data_dir, triplets_per_user=100, transform=None):
        """
        Args:
            base_data_dir (str): The root directory of the signature dataset.
            triplets_per_user (int): The number of triplets to generate per user (virtual epoch size).
            transform (callable, optional): PyTorch transforms to apply to images.
        """
        self.base_data_dir = base_data_dir
        self.triplets_per_user = triplets_per_user
        self.transform = transform
        self.signature_map = self._create_signature_map()
        self.user_ids = list(self.signature_map.keys())
        if not self.user_ids:
            raise RuntimeError(f"No valid individuals found in {base_data_dir}. Check directory structure and image counts.")

    def _create_signature_map(self):
        real_signatures_dir = os.path.join(self.base_data_dir, 'Real')
        fake_signatures_dir = os.path.join(self.base_data_dir, 'Fake')
        signature_map = defaultdict(lambda: {'real': [], 'fake': []})
        if not os.path.isdir(real_signatures_dir):
            raise FileNotFoundError(f"Error: Directory not found at {real_signatures_dir}")
        if not os.path.isdir(fake_signatures_dir):
            raise FileNotFoundError(f"Error: Directory not found at {fake_signatures_dir}")
        all_ids = sorted(os.listdir(real_signatures_dir))
        for user_id in all_ids:
            if user_id.startswith('ID_'):
                real_images = glob.glob(os.path.join(real_signatures_dir, user_id, '*.jpg'))
                fake_images = glob.glob(os.path.join(fake_signatures_dir, user_id, '*.jpg'))
                if len(real_images) >= 2 and len(fake_images) >= 1:
                    signature_map[user_id]['real'] = real_images
                    signature_map[user_id]['fake'] = fake_images
        return signature_map

    def __len__(self):
        return len(self.user_ids) * self.triplets_per_user

    def __getitem__(self, index):
        person_id = random.choice(self.user_ids)
        anchor_path, positive_path = random.sample(self.signature_map[person_id]['real'], 2)
        negative_path = random.choice(self.signature_map[person_id]['fake'])
        anchor_img = self._load_image(anchor_path)
        positive_img = self._load_image(positive_path)
        negative_img = self._load_image(negative_path)
        return (anchor_img, positive_img, negative_img)

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

    mean = [0.861, 0.861, 0.861]
    std = [0.274, 0.274, 0.274]
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=0, shear=10, translate=(0.1,0.1)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    embedding_dim = 128
    embedding_net = SimpleEmbeddingNetwork(embedding_dim=embedding_dim)
    siamese_network = SiameseNetwork(embedding_network=embedding_net)

    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    optimizer_siamese = optim.AdamW(siamese_network.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer_siamese, step_size=2, gamma=0.1)