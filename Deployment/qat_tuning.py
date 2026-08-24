import torch
import torch.nn as nn
import torch.ao.quantization as quantization


class BaseCNN(nn.Module):

    def __init__(self):
        super().__init__()
        # conv + batchNorm
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        # max pooling
        self.pool = nn.MaxPool2d(2, 2)
        # dropout
        self.dropout = nn.Dropout(0.2)
        # fc
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, 10)
        # relu
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        # flatten
        x = x.view(-1, 512 * 2 * 2) 
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x) 
        return x

class QATCNN(nn.Module):

    def __init__(self):
        super().__init__()
        # just like PTQ, we need to add QuantStub and DeQuantStub
        self.quant = quantization.QuantStub()
        self.dequant = quantization.DeQuantStub()

        # Caution: 
        # for layer fusion, ReLU should be defined as separate layers
        
        # conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)

        # conv2
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        # conv3
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=True)

        # shared layers
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.2)
        
        # FC
        self.fc1 = nn.Linear(512 * 2 * 2, 512)
        self.relu_fc = nn.ReLU(inplace=True)

        # classification Layer
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # quantize
        x = self.quant(x)

        x = self.relu1(self.bn1(self.conv1(x)))  # to fuse
        x = self.pool(x)
        x = self.relu2(self.bn2(self.conv2(x)))  # to fuse
        x = self.pool(x)
        x = self.relu3(self.bn3(self.conv3(x)))  # to fuse
        x = self.pool(x)

        x = x.reshape(-1, 512 * 2 * 2)
        
        x = self.relu_fc(self.fc1(x))  # to fuse
        x = self.dropout(x)
        x = self.fc2(x)

        # dequantize
        x = self.dequant(x)
        return x

    def fuse_model(self):
        # combines (Conv, BN, ReLU) into a single block
        quantization.fuse_modules(self, ['conv1', 'bn1', 'relu1'], inplace=True)
        quantization.fuse_modules(self, ['conv2', 'bn2', 'relu2'], inplace=True)
        quantization.fuse_modules(self, ['conv3', 'bn3', 'relu3'], inplace=True)
        
        # combines (FC, ReLU) into a single block
        quantization.fuse_modules(self, ['fc1', 'relu_fc1'], inplace=True)




if __name__ == "__main__":

    baseline_model_path = "./base_model.pt"
    # base_model = BaseCNN()
    baseline_model_state_dict = torch.load(baseline_model_path, weights_only=True)

    qat_model = QATCNN()
    # if the structure or layer names of the old model is different from the QAT model,
    # then we should mapping old weights to the new ones.
    # but in this case, we are the same with new model, so we omit this mappings.
    qat_model.load_state_dict(baseline_model_state_dict, strict=False)

    # layer fusion
    qat_model.eval()
    qat_model.fuse_model()
    qat_model.qconfig = quantization.get_default_qat_qconfig('qnnpack') # for arm

    # set back to train mode
    qat_model.train()
    qat_model = quantization.prepare_qat(qat_model)

    # training the model
    ...

    # move the model to CPU
    qat_model.to("cpu")

    # convert
    final_quantized_model = quantization.convert(qat_model.eval(), inplace=False)

