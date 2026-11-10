# MNIST Classifier

This exercise demonstrates a basic PyTorch image-classification workflow with
the MNIST handwritten-digit dataset. It defines a custom fully connected
classifier, preprocesses grayscale images, trains with mini-batches, and
measures accuracy on the MNIST test split.

## Workflow

- `MyMNISTClassifier` flattens each 28 x 28 image and passes its 784 pixel values through a 128-unit ReLU layer and a 10-class output layer.
- Images are converted to tensors and normalized with the commonly used MNIST mean (`0.1307`) and standard deviation (`0.3081`).
- The training split uses batches of 64 with shuffling; the test split uses batches of 1,000 without shuffling.
- The model is trained for two epochs per cycle, over three cycles. A single Adam optimizer is retained across cycles so its state persists as training continues.
- Test accuracy is printed after each cycle. For a strict final evaluation, avoid using test results to tune training choices; use a separate validation split for that purpose.
