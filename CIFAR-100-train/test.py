from torch.utils.data import Dataset
from PIL import Image
import os
import scipy
import torch
import torch.nn as nn

dir = "E:\\牛津大学Flowers102数据集"

# label_mat = scipy.io.loadmat(os.path.join(dir, 'imagelabels.mat'))
# labels = label_mat["labels"][0]
# print(f"max : {labels.max()}")
# print(f"min : {labels.min()}")

# class MyDataset(Dataset):

#     def __init__(self) -> None:
#         super().__init__()
#         self.img_path = os.path.join(dir, "102flowers","jpg")
#         label_mat = scipy.io.loadmat(os.path.join(dir, 'imagelabels.mat'))
#         self.labels = label_mat["labels"][0] - 1

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, index):
#         img_name = f"image_{index+1:05d}.jpg"
#         img = Image.open(os.path.join(self.img_path, img_name))
#         return img, self.labels[index]
    
# image_sets = MyDataset()
# for i in range(580,681,5):
#     img,label = image_sets[i]
#     print(f"image {i} : {img.size}  label: {label}")

class MyModel(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.layer3 = SubModule()
    
    def forward(self, x):
        return self.layer3(x)

class SubModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(3,5),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(5,1)
        )
        self.layer2 = nn.Conv1d(1,3,kernel_size=3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# model = nn.Sequential(
#     nn.Conv1d(1,3, kernel_size=3, padding=1, stride=1),
#     nn.ReLU(),
#     nn.Flatten(),
#     nn.Linear(6,10),
#     nn.ReLU(),
#     nn.Linear(10,2)
# )

model = MyModel()
# print(model)
[print(param) for param in model.named_parameters()]

