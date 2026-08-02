import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torchvision import transforms
from collections import defaultdict
import glob
import os
import sys
import random
import re
from pathlib import Path
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
        self.triplets_per_user = triplets_per_user
        self.transform = transform
        self.signature_map = utils.create_signature_map(base_data_dir)
        self.user_ids = list(self.signature_map.keys())

    def __len__(self):
        # custom defined length
        return len(self.user_ids) * self.triplets_per_user

    def __getitem__(self, index):
        # custom defined __getitem__ without consideration of index
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


def verify_signature(model, anchor_path, verify_path, val_transform, threshold=0.7):

    # saved_file = Path(__file__).parent / saved_filename
    # model.load_state_dict(torch.load(saved_file))
    
    try:
        # load signatures
        sig_anchor = val_transform(Image.open(anchor_path).convert("RGB")).unsqueeze(0)
        sig_verify = val_transform(Image.open(verify_path).convert("RGB")).unsqueeze(0)
    except FileNotFoundError as e:
        print(f"Error loading signatures: {e}")
        return
    
    model.eval() 
    with torch.no_grad():
        emb_genuine, emb_test = model(sig_anchor, sig_verify, triplet_bool=False)

        # Calculate the euclidean distance between the two embeddings.
        distance = F.pairwise_distance(emb_genuine, emb_test).item()
        
        # Make a prediction based on whether the distance is below the threshold.
        is_genuine = distance < threshold
        
    print(f"--- Verification Result ---")
    print(f"Distance: {distance:.4f}")
    print(f"Decision Threshold: {threshold:.4f}")

    if is_genuine:
        print("Prediction: ✅ Genuine Signature\n")
    else:
        print("Prediction: ❌ Forgery Detected\n")

    
    # Create a figure with two subplots for visual comparison.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 3))
    # Set the main title to distance.
    fig.suptitle(f'Distance: {distance:.4f}', fontsize=14)
    
    # anchor in the left
    ax1.imshow(Image.open(anchor_path))
    ax1.set_title("Anchor")
    ax1.axis('off')
    
    # verified in the right
    ax2.imshow(Image.open(verify_path))
    ax2.set_title("Genuine" if is_genuine else "Forged")
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()
     

if __name__ == "__main__":
    
    saved_filename = "sig_siamese.pth"
    current_dir = Path(__file__).parent
    signature_base_dir = ""  # <<---- change this line to your own data path
    
    mean = [0.861, 0.861, 0.861]
    std = [0.274, 0.274, 0.274]
    train_transform = transforms.Compose([
        # transforms.RandomAffine(degrees=0, shear=10, translate=(0.1,0.1)),
        # transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    val_transform = transforms.Compose([
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # ==== training progress ====
    n_epochs = 10
    embedding_dim = 128
    embedding_net = SimpleEmbeddingNetwork(embedding_dim=embedding_dim)
    model = SiameseNetwork(embedding_network=embedding_net)

    triplet_loss = nn.TripletMarginLoss(margin=1.5, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    dataset:Dataset = SignatureTripletDataset(base_data_dir=signature_base_dir, triplets_per_user=10, transform=train_transform)
    dataloader:DataLoader = DataLoader(dataset, batch_size=24, shuffle=True)

    model.train()
    training_lowest_loss = None
    for epoch in range(1, n_epochs + 1):
        running_train_loss = 0.0
        for triplets, _  in dataloader:        
            anchors, positives, negatives = triplets
            optimizer.zero_grad()
            anchor_outs, positive_outs, negative_outs = model(anchors, positives, negatives)
            loss = triplet_loss(anchor_outs, positive_outs, negative_outs)
            
            loss.backward()
            optimizer.step()
            
            # Weight loss by batch size for correct averaging
            batch_size = anchors.size(0)
            running_train_loss += loss.item() * batch_size
        
        train_loss = running_train_loss / len(dataset)

        # summary for the epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")

        # save the model parameters when it is the lowest loss
        if not training_lowest_loss or training_lowest_loss > train_loss:
            training_lowest_loss = train_loss
            torch.save(model.state_dict(), current_dir / saved_filename)
            print(f"==== Save model at loss of {train_loss}")

        # Update the learning rate scheduler, if one is provided
        if scheduler:
            scheduler.step()


    # ==== Evaluation =====
    # saved_file = Path(__file__).parent / saved_filename
    # model.load_state_dict(torch.load(saved_file))
    # anchor_img_path = os.path.join(signature_base_dir, "DigitalReal", "digital_real_114_3.jpg")
    # verified_img_path = os.path.join(signature_base_dir, "DigitalFake", "digital_fake_27_3.jpg")
    # verify_signature(model, anchor_img_path, verified_img_path, val_transform, threshold=0.7)
    

    # ==== Test code for preview the triplets ====
    # fig, axses = plt.subplots(nrows=3, ncols=3)
    # for i in range(axses.shape[0]):
    #     triplet_file, triplet_filename = dataset[0]
    #     for j in range(axses.shape[1]):
    #         axs = axses[i, j]
    #         axs.imshow(triplet_file[j])
    #         axs.axis('off')
    #         axs.set_title(triplet_filename[j])
    # plt.tight_layout()
    # plt.show()