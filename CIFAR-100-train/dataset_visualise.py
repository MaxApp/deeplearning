
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":

    # 使用预下载好的训练集，加载至 dataset
    # 随机重中抽取图片数据进行抽样可视化展示
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR100("E:\\CIFAR100分类数据集", train=True, download=False, transform=transform)

    # 随机取 12 张图片 3x4 显示
    fig, axes = plt.subplots(3,4, figsize=(4,4))
    indecies = torch.randint(1,100, (1,12)).squeeze().numpy()
    for i, idx in enumerate(indecies):
        # print(f"i:{i//4} j:{i%4} idx:{idx}")
        img, label = dataset[idx]
        m,n = i//4, i%4
        axes[m,n].imshow(img.permute(1, 2, 0))
        axes[m,n].set_title(f'{dataset.classes[label]}', fontsize=6)
        axes[m,n].axis('off')
        
    plt.tight_layout()  # 自动调整子图间距
    plt.show()




