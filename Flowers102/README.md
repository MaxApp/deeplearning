# Oxford Flowers102 Dataset Preparation

This exercise introduces custom PyTorch datasets and image preprocessing using
the Oxford Flowers102 image-classification dataset. It demonstrates how to
load images and labels, apply transforms, create reproducible train/validation/
test splits, and batch samples with `DataLoader`.

## Dataset layout

Download and extract the dataset, then arrange the files like this:

```text
<dataset-root>/
├── imagelabels.mat
└── 102flowers/
    └── jpg/
        ├── image_00001.jpg
        ├── image_00002.jpg
        └── ...
```

The script expects this layout: image paths are built under
`<dataset-root>/102flowers/jpg`, while labels are loaded from
`<dataset-root>/imagelabels.mat`.

## What the exercise demonstrates

- `FlowersDataset` loads an image and its corresponding MATLAB label.
- Original labels are 1-based; the dataset converts them to zero-based Python
  class indices (`0` through `101`), which are suitable for PyTorch
  classification losses.
- Training images receive random horizontal flips, rotations, and brightness
  changes. Validation and test images use deterministic resizing and cropping
  without random augmentation.
- The dataset is split into 70% training, 15% validation, and 15% test data.
  A fixed generator seed makes the split reproducible.
- `DataLoader` batches samples, the training loader shuffles its samples, while
  validation and test loaders preserve their split order.

The script prints the dataset sizes and inspects one training batch. It is a
preparation example, not a model-training or evaluation script.
