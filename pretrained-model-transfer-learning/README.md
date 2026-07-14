# Pretrained models in Pytorch in action

By the power of `torchvision` library, we can make use of most famous models effortlessly into projects.
We can do inference directly with pre-trained parameters or we can do transfer learning and fine-tuning with them to adapter to our cases.
Here we inspect the modules to understand their architecture and make some practicals in object prediction, segmentation and detection with some of pre-trained models.

## resnet50 prediction


## deeplabv3_resnet50_segmentation.py

Based on `DeepLabV3` model, we do image segmentations by `torchvision.utils.draw_segmentation_masks`.

In order to make sure the items model supported, we read classes from weights meta:

```text
['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
```

There're total 21 classes including `__background__`. We choose a horse to do the experiment.

Supply a custom image which contains a horse, define the target object and mask colors, convert it to tensor and feed into the model to get individual pixel probabilities and calculate the mask.

Comparision shows below:

![masked_horses](imgs/horse1_masked.png)



## faster R-CNN resnet50 FPN detection
