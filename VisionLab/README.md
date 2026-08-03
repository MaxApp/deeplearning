# Vision Specialized Laboratory

This is the experimental sections about torch vision. We'll hands on some simple labcodes to get an intuition about convolutional network.

### Feature Maps

What's the output of the conv layer? What do they do after images pass through a conv layer and max pool? How do filters affect the results? We'll retrieve the output of a layer and display every channel using plot, that's called `feature maps`.

#### feature_map.py

```python
# read the image
image = Image.open(current_dir / "imgs/bird.jpg")
# convert to tensor
image_tensor = transforms.ToTensor()(image)

# display
# utils.plot_image(image_tensor)

# conv block
conv_block = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=16, out_channels=9, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2)
)
# apply the convolutional layer to the image
output_conv_layer = conv_block(image_tensor)

utils.plot_channels(output_conv_layer)
```

![feature_map](imgs/feature_maps_1.png)


