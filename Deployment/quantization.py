import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub, get_default_qconfig
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets


class MyCNN(nn.Module):

    def __init__(self):
        super().__init__()
        # conv + batchNorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout = nn.Dropout(0.2)
        # fc
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        # relu
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        # flatten
        x = x.view(-1, 512 * 2 * 2) 
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x) 
        return x

class MyQuantizedCNN(nn.Module):

    def __init__(self):
        super().__init__()
        # quant stub
        self.quant = QuantStub()

        # conv + batchNorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout = nn.Dropout(0.2)
        # fc
        self.fc1 = nn.Linear(512 * 2 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)
        # relu
        self.relu = nn.ReLU()

        # dequant stub
        self.dequant = DeQuantStub()


    def forward(self, x):
        x = self.quant(x)

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        # flatten
        x = x.view(-1, 512 * 2 * 2) 
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x) 

        x = self.dequant(x)

        return x

def calibrate(model, data_loader, num_batches=20):
    """calibrate quantization progress"""
    model.eval()
    
    total_batches = len(data_loader)
    num_calibration_batches = min(num_batches, total_batches)

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            # only loop for a small batches
            if i >= num_calibration_batches:
                break
            model(images)


if __name__ == "__main__":
    # constant values
    # mean, std data for CIFAR-10 in advance
    cifar10_mean = (0.5071, 0.4867, 0.4408)
    cifar10_std = (0.2675, 0.2565, 0.2761)

    target_classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Training set transform
    train_transform = transforms.Compose([
        # Data augmentation
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.RandomRotation(15), #  [-15,15]

        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    # Validation set transform
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])

    # load pre-downloaded dataset
    data_path = "./CIFAR10"
    original_train_dataset = datasets.CIFAR10(data_path, train=True, download=False, transform=train_transform)
    original_val_dataset = datasets.CIFAR10(data_path, train=False, download=False, transform=val_transform)

    batch_size = 64
    # data loader for the training set, with shuffling enabled
    train_loader = DataLoader(original_train_dataset, batch_size=batch_size, shuffle=True)
    # validation loader for test, with shuffling disabled
    val_loader = DataLoader(original_val_dataset, batch_size=batch_size, shuffle=False)

    # pretrained parametes
    model_pt_file = "./cifar10_30_epochs_best.pt"
    baseline_model = MyCNN()
    baseline_model.load_state_dict(torch.load(model_pt_file))
    baseline_model.eval()

    # check layer weights types
    for name, module in baseline_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"Layer: {name} | Weight dtype: {module.weight.dtype}")

    # an new instance of model
    quantized_static_model = MyQuantizedCNN()
    # copy parameters from base
    quantized_static_model.load_state_dict(baseline_model.state_dict())
    quantized_static_model.eval()
    quantized_static_model.qconfig = get_default_qconfig('x86')
    torch.ao.quantization.prepare(quantized_static_model, inplace=True)

    # calibrate data
    calibrate(quantized_static_model, val_loader)
    # convert to quantized model
    torch.ao.quantization.convert(quantized_static_model, inplace=True)

    # check layer weights types
    for name, module in quantized_static_model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            print(f"Layer: {name} | Weight dtype: {module.weight.dtype}")


