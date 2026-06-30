# import torch
# import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import transforms
import scipy
import os
from PIL import Image

# Custom Dataset
class FlowersDataset(Dataset):
    def __init__(self, root_path, transform):
        self.transform = transform
        self.imgs_root = os.path.join(root_path,'102flowers','jpg')
        self.labels_mat = scipy.io.loadmat(os.path.join(root_path, 'imagelabels.mat'))
        self.labels = self.labels_mat['labels'].squeeze() - 1  # 所有 label - 1

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        image_name = os.path.join(self.imgs_root, f"image_{idx+1:05d}.jpg")
        image = Image.open(image_name)
        image_tensor = self.transform(image)
        return image_tensor, self.labels[idx]
    

if __name__ == "__main__":
    root_path = "E:\\牛津大学Flowers102数据集"

    # transform
    transform = transforms.Compose([
        # 图片增强
        transforms.RandomHorizontalFlip(p=0.5),  # 随机翻转
        transforms.RandomRotation(degrees=10),  # 随机旋转
        transforms.ColorJitter(brightness=0.2), # 随机亮度

        # 图片裁剪
        transforms.Resize(256),
        transforms.CenterCrop(224),
        # 转向量
        transforms.ToTensor(),
        # 标准化
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = FlowersDataset(root_path, transform)

    print(f"total images : {len(dataset)}")

    # split dataset
    train_set_count = int(len(dataset) * 0.70)
    val_set_count = int(len(dataset) * 0.15)
    test_set_count = len(dataset) - train_set_count - val_set_count
    train_set, val_set, test_set = random_split(
        dataset, [train_set_count, val_set_count, test_set_count]
    )

    # train data
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    for imgs, labels in train_loader:
        print(f"image tensors: {imgs.shape}, \nlabels: {labels}")
        break

    # print out
    print(f"Train: total->{train_set_count},  batches->{len(train_loader)}")
    print(f"Val: total->{val_set_count}")
    print(f"Test: total->{test_set_count}")
