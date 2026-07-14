import torch
import torchvision
from torchvision import models as tv_models
import torchvision.transforms.v2 as transforms
from pathlib import Path


if "__main__" == __name__:
    weights = tv_models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    # detection_model = tv_models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights).eval()

import torchvision.models as tv_models
model = tv_models.detection.fasterrcnn_resnet50_fpn(weights=None)