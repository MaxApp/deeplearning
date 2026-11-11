# EMNIST Letters Classifier

This script demonstrates a basic PyTorch classification workflow using the letters split of the EMNIST dataset. It covers loading a torchvision dataset, normalizing grayscale images, building a fully connected neural network, training with mini-batches and evaluating letter classification accuracy. Like the [MNIST example](../MNIST-train/README.md), it uses a simple fully connected network rather than a convolutional neural network (CNN).



## Workflow

- The script loads the `letters` split for both training and testing.
- Images are converted to tensors and normalized with mean `0.1736` and standard deviation `0.3317`.
- The model flattens each 28 x 28 image, uses two fully connected ReLU hidden layers and outputs 26 class scores.
- EMNIST Letters labels are 1-based. The training and evaluation loops subtract one so targets use class indices `0` through `25` as required by `CrossEntropyLoss`.
- The model is trained for 10 epochs with batches of 64 using Adam.
- Training loss is averaged by sample count including a potentially smaller final batch.
