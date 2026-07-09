import torchvision.utils as vutils
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

mean = (0.1736,)
std = (0.3317,)
fake_transforms = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)]
)
images = datasets.FakeData(10, transform=fake_transforms)
dataloader = DataLoader(images, shuffle=False, batch_size=6)

for imgs,lables in dataloader:
    grid =vutils.make_grid(imgs, nrow=3, normalize=True, padding=6)

    # 将张量转为numpy格式以便显示
    grid_np = grid.permute(1, 2, 0).numpy()  # 从(C,H,W)转为(H,W,C)
    
    # 显示图像
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_np)
    plt.axis('off')  # 隐藏坐标轴
    plt.title('Fake Data Grid Visualization')
    plt.show()
