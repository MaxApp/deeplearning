# Flexible CNN model
Dynamically constructure blocks of modules to build a flexible network.
Facilitate the process of tuning hyperparameters, in order to find an optimal one.

## Architecture

Input layer for original image input features, followed by customized conv layer, then flattened, flow data to customized fc layers, finally to classification layer.

```text
Input Layer
|
Convolutional Layers   x n
|
Flatten Layer
|
FC Layers   x n
|
Linear Output Layer
```

### Building blocks

* ConvBlock
* FCBlock
* Flatten
* Classify Block

### Cautions
1. Do not use python `list` for dynamically reserve modules, use `nn.ModuleList` instead, or parameters won't be registered and updated during training.
2. It's better to calculate the flatten size according to input layer features in `__init__` method. Using an internel method for conv calculation.
3. Print out the models to confirm the structure is what you expect.
