# Pretrained models in Pytorch in action

By the power of `torchvision` library, we can make use of most famous models effortlessly into projects.
We can do inference directly with pre-trained parameters or we can do transfer learning and fine-tuning with them to adapter to our cases.
Here we inspect the modules to understand their architecture and make some practicals in object prediction, segmentation and detection with some of pre-trained models.

## Scripts Content Introduction

### resnet50_prediction.py

We use `Resnet50` model to predict image content. Instead of using `pretrained=True` parameter to download file online instantly, we try to pre-download `resnet50-0676ba61.pth` and `imagenet_class_index.json` first and load them offline manually.

With the model ready, we provide images and do transforms to make predictions, verify the ability of the model.
Also we need to denormalize the cropped images and matching the category_id with class names, eventually show them up to visualize the results.

![prediction_results](imgs/resnet50_prediction.png)


### deeplabv3_resnet50_segmentation.py

Based on `DeepLabV3` model, we do image segmentations by `torchvision.utils.draw_segmentation_masks`.

In order to make sure the items model supported, we read classes from weights meta:

```text
['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

There're total 21 classes including `__background__`. We choose a horse to do the experiment.

Supply a custom image which contains a horse, define the target object and mask colors, convert it to tensor and feed into the model to get individual pixel probabilities and calculate the mask.

Comparision shows below:

![masked_horses](imgs/horse1_masked.png)


### faster_rcnn_resnet50_detection.py

Use the power of `Fast R-CNN` model to perform object detection. The popular model combines components:

* Faster R-CNN (Region-based Convolutional Neural Network)
* ResNet-50
* FPN (Feature Pyramid Network): an effective multi-scale detection strategy


