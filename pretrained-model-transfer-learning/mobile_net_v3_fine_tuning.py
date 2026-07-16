import torch
import torch.nn as nn
import torchvision.models as tv_models
import torchvision.transforms as transforms
from pathlib import Path

if __name__ == "__main__":

    # load model with pretrained weights
    mobilenet_model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # freeze pretrained parameters for feature layers
    for feature_parameter in mobilenet_model.features.parameters():
        feature_parameter.requires_grad = False

    # the final classification layer
    last_classifier_layer = mobilenet_model.classifier[-1]
    # the in_features of last_classifier_layer
    num_features = last_classifier_layer.in_features

    # our target classifications for MINIST
    num_classes = 26

    # substitude for a new layer
    mobilenet_model.classifier[-1] = nn.Linear(in_features=num_features, out_features=num_classes)

    # training strategy
    loss_function = torch.nn.CrossEntropyLoss()
    # Only optimize the parameters that require gradients
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, mobilenet_model.parameters()), lr=0.001)
