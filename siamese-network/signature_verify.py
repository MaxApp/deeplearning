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
        # custom defined getitem without consideration of index
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


def verify_signature(genuine_path, test_path, threshold=0.7):

    mean = [0.861, 0.861, 0.861]
    std = [0.274, 0.274, 0.274]
    val_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    embedding_dim = 128
    embedding_net = SimpleEmbeddingNetwork(embedding_dim=embedding_dim)
    model = SiameseNetwork(embedding_network=embedding_net)
    saved_file = Path(__file__).parent / saved_filename
    model.load_state_dict(torch.load(saved_file))
    
    try:
        img_genuine = val_transform(Image.open(genuine_path).convert("RGB")).unsqueeze(0)
        img_test = val_transform(Image.open(test_path).convert("RGB")).unsqueeze(0)
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        return
    
    model.eval() 
    with torch.no_grad():
        emb_genuine = model.get_embedding(img_genuine)
        emb_test = model.get_embedding(img_test)
        
        # Calculate the euclidean distance between the two embeddings.
        distance = F.pairwise_distance(emb_genuine, emb_test).item()
        
        # Make a prediction based on whether the distance is below the threshold.
        is_genuine = distance < threshold
        
    # Display the numerical results of the verification.
    print(f"--- Verification Result ---")
    print(f"Distance: {distance:.4f}")
    print(f"Decision Threshold: {threshold:.4f}")
    # Print the final prediction outcome.
    if is_genuine:
        print("Prediction: ✅ Genuine Signature\n")
    else:
        print("Prediction: ❌ Forgery Detected\n")

    """
    # Create a figure with two subplots for visual comparison.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    # Set the main title of the figure to show the calculated distance.
    fig.suptitle(f'Distance: {distance:.4f}', fontsize=16)
    
    # Display the known genuine signature in the left subplot.
    ax1.imshow(Image.open(genuine_path))
    ax1.set_title("Known Genuine Signature")
    ax1.axis('off')
    
    # Display the signature to be verified in the right subplot.
    ax2.imshow(Image.open(test_path))
    ax2.set_title("Signature to Verify")
    ax2.axis('off')
    
    # Render the final plot.
    plt.show()
    """ 

if __name__ == "__main__":
    saved_filename = "sig_siamese.pth"
    current_dir = Path(__file__).parent
    signature_base_dir = "E:\\PDF\\pytorch\\C3M1\\signatures_train"

    # """ Verfification """
    img_real = os.path.join(signature_base_dir, "DigitalReal", "digital_real_114_3.jpg")
    img_fake = os.path.join(signature_base_dir, "DigitalFake", "digital_fake_27_3.jpg")
    verify_signature(genuine_path=img_real, test_path=img_fake)


    '''
    
    mean = [0.861, 0.861, 0.861]
    std = [0.274, 0.274, 0.274]
    train_transform = transforms.Compose([
        # transforms.RandomAffine(degrees=0, shear=10, translate=(0.1,0.1)),
        # transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        # transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    

    n_epochs = 10
    embedding_dim = 128
    embedding_net = SimpleEmbeddingNetwork(embedding_dim=embedding_dim)
    model = SiameseNetwork(embedding_network=embedding_net)

    triplet_loss = nn.TripletMarginLoss(margin=1.5, p=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    dataset:Dataset = SignatureTripletDataset(base_data_dir=signature_base_dir, triplets_per_user=10, transform=train_transform)
    dataloader:DataLoader = DataLoader(dataset, batch_size=24, shuffle=True)

    # ==== Training ====
    model.train()
    lowest_loss = None
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
            print(f"---> batch size: {batch_size}")
            running_train_loss += loss.item() * batch_size
        
        train_loss = running_train_loss / len(dataset)

        # # --- Validation Phase ---
        # model.eval()
        # correct_predictions = 0
        # total_pairs = 0
        # running_val_loss = 0.0
        # val_samples_processed = 0
        # val_progress_bar = tqdm(val_loader, desc=f"Epoch {epoch}/{n_epochs} [Validation]", leave=False)
        
        # with torch.no_grad():
        #     for data_batch in val_progress_bar:
        #         anchors, positives, negatives = data_batch
        #         anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                
        #         anchor_out, pos_out, neg_out = model(anchors, positives, negatives)
        #         val_loss_item = loss_fcn(anchor_out, pos_out, neg_out)

        #         # Weight loss by batch size for correct averaging
        #         batch_size = anchors.size(0)
        #         running_val_loss += val_loss_item.item() * batch_size
        #         val_samples_processed += batch_size

        #         # Accuracy calculation
        #         dist_pos = F.pairwise_distance(anchor_out, pos_out)
        #         correct_predictions += torch.sum(dist_pos < threshold).item()
        #         dist_neg = F.pairwise_distance(anchor_out, neg_out)
        #         correct_predictions += torch.sum(dist_neg >= threshold).item()
        #         total_pairs += len(dist_pos) + len(dist_neg)

        #         # Update running metrics on the progress bar
        #         current_acc = correct_predictions / total_pairs if total_pairs > 0 else 0
        #         display_loss = running_val_loss / val_samples_processed
        #         val_progress_bar.set_postfix(acc=f'{current_acc:.2%}', loss=f'{display_loss:.4f}')

        # val_accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0
        # val_loss = running_val_loss / len(val_loader.dataset)

        # Print a summary for the epoch
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | LR: {current_lr:.6f}")

        # save the lowest loss model parameter
        if not lowest_loss or lowest_loss > train_loss:
            lowest_loss = train_loss
            torch.save(model.state_dict(), current_dir / saved_filename)
            print(f"==== save loss: {train_loss}")

        # Update the learning rate scheduler, if one is provided
        if scheduler:
            scheduler.step()
    '''
            
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


    """
    # ==== Evaluation =====
    img_real = os.path.join(signature_base_dir, "DigitalReal", "digital_real_114_3.jpg")
    img_fake = os.path.join(signature_base_dir, "DigitalFake", "digital_fake_27_3.jpg")
    val_transform = transforms.Compose([
            # transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
    ])

    try:
        img_genuine = val_transform(Image.open(img_real).convert("RGB")).unsqueeze(0)
        img_test = val_transform(Image.open(img_fake).convert("RGB")).unsqueeze(0)
    except FileNotFoundError as e:
        print(f"Error loading image: {e}")
        sys.exit()

    threshold = 0.7
    model.eval() 
    with torch.no_grad():

        emb_genuine = model.get_embedding(img_genuine)
        emb_test = model.get_embedding(img_test)
        
        # Calculate the euclidean distance between the two embeddings.
        distance = F.pairwise_distance(emb_genuine, emb_test).item()
        
        # Make a prediction based on whether the distance is below the threshold.
        is_genuine = distance < threshold
        
    # Display the numerical results of the verification.
    print(f"--- Verification Result ---")
    print(f"Distance: {distance:.4f}")
    print(f"Decision Threshold: {threshold:.4f}")
    # Print the final prediction outcome.
    if is_genuine:
        print("Prediction: ✅ Genuine Signature\n")
    else:
        print("Prediction: ❌ Forgery Detected\n")
    """