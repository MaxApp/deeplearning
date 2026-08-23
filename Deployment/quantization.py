import torch
import torch.nn as nn

class QuantizedCNN(nn.Module):

    def __init__(self):
        super(QuantizedCNN, self).__init__()
        self.quant = torch.quantization.QuantStub()
        ... # Model Layers
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        ... # Model Layers
        x = self.dequant(x)
        return x

