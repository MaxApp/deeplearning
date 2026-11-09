# Fundamentals and Theories

The fundamentals and basic concepts in machine learning with experiments and implementations. These are the pivots of next step for the real tasks to train a model. With a solid foundation in conception we could go further during the journey.

## Overview of Machine Learning

Most common categories of machine learning we used includes:

* Supervised
    * regression model
    * classification model
* Unsupervised
    * clustering
    * anomaly detection
    * dimensionality reduction
* Reinforcement Learning

## Concepts

The basic concepts we should be familiar with before hands on.

* linear regression
* logistic regression
* cost function
* gradient descent
* learning rate
* vectorization
* normalization
* regularization
* neural networks architectures
* forward and backward propagation
* activation functions
* clustering with K-means
* anomaly detection with normal distribution
* collaborative filtering
* content based filtering
* PCA
* state-actoin value function
* 

## Basic Trial 

### simple_linear_dynamic.py

This script demonstrates the complete training loop for fitting a simple
single-variable linear model. It creates synthetic observations from the
relationship `y = w * x + b`, adds random noise, and trains a PyTorch
`nn.Linear` layer to recover the underlying weight and bias.

The script also updates a Matplotlib plot while training. Watching the
predicted line move toward the noisy sample data makes the gradual effect of
gradient descent easier to understand.

Basic concepts used:
1. `nn.Linear` as a simple model
2. `torch.linspace` for generating input values
3. `unsqueeze` for reshaping
4. `torch.rand` for generating random noise
5. `nn.MSELoss` for loss function
6. `optim.SGD` for gradient descent
7. `loss.backward()` for backward propagation  
8. `optimizer.step()` for updating

Visualization example

![train1](imgs/simple_linear_dynamic_01.png)

### simple_curve_dynamic.py

This script demonstrates nonlinear regression by fitting noisy `y = log10(x)`
data with a multi-layer neural network. It highlights the importance of
`BatchNorm1d` for normalization and compares `Adam` with `SGD` during training.

Key features:
1. Multi-layer network with `nn.Linear` and `nn.ReLU`
2. Input and hidden-layer normalization with `BatchNorm1d`
3. MSE loss and Adam/SGD optimizer comparison
4. Loss tracking and dynamic Matplotlib visualization

Visualization example:

![train1](imgs/simple_curve_dynamic_01.png)

### residual_network.py

This script fits the linear relationship `y = 2x + 1` with a PyTorch network
that includes a residual block. The block adds its input to the output of two
linear layers, demonstrating how a skip connection works when the feature
dimensions match.

### dropout_exp.py

This example builds a two-layer network with a manual dropout layer. During training, it creates a Bernoulli dropout filter for
the hidden activations, keeps each activation with probability `0.8`, and
divides kept values by `0.8` (inverted dropout) so their expected magnitude is
preserved. Dropout is disabled during evaluation.

## Notes

### Momentum, RMSprop and Adam

Let $g$ be the current gradient and $\eta$ the learning rate. The formulas
below use elementwise operations; $\epsilon$ is a small constant.

- **Momentum** updates a running direction:
  $v_t = \mu v_{t-1} + g_t$; $\theta_{t+1} = \theta_t - \eta v_t$.
  Its main effect is to smooth and build up the update direction across steps.
  It uses the same learning rate for every parameter.
- **RMSprop** tracks squared-gradient size:
  $s_t = \rho s_{t-1} + (1-\rho)g_t^2$;
  $\theta_{t+1} = \theta_t - \eta g_t / (\sqrt{s_t} + \epsilon)$.
  Its main effect is to adapt the step size per parameter: larger recent
  gradients mean smaller steps.
- **Adam** tracks both gradient direction and squared-gradient size:
  $m_t = \beta_1m_{t-1} + (1-\beta_1)g_t$;
  $v_t = \beta_2v_{t-1} + (1-\beta_2)g_t^2$;
  $\theta_{t+1} = \theta_t - \eta \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$.
  Its main effect is to combine Momentum's smoothed direction with RMSprop's
  per-parameter step sizes. The hatted values are bias-corrected averages,
  compensating for their initial bias toward zero.

**Main difference:** Momentum changes the update direction using past
gradients; RMSprop changes each parameter's step size using recent gradient
magnitudes; Adam does both.

### Input normalization, Batch normalization and Layer normalization

These methods keep values on a useful scale, which can make training more
stable, but they normalize different data:

- **Input normalization** is applied to the input features before they enter the
  model. For example, scale pixel values to `[0, 1]` or standardize each feature
  using training-set statistics. Apply the same transformation at inference.
- **Batch normalization** normalizes each feature (or channel) using statistics
  from the current mini-batch during training. It keeps running statistics for
  inference and has learnable scale and shift parameters. Its batch-dependent
  statistics can be less suitable for very small or variable batch sizes.
- **Layer normalization** normalizes the features within each individual
  example, rather than across the batch. It does not depend on batch size and is
  commonly used in Transformer models. It also has learnable scale and shift
  parameters.

**In short:** input normalization transforms raw data; batch normalization
normalizes across examples in a batch; layer normalization normalizes within
each example.
