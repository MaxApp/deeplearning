# Decision Tree and Tree Ensembles

Tree-based models make predictions by splitting data into groups according to feature values. A decision tree uses a sequence of these splits, while ensembles such as `random forests` and `gradient-boosted trees` combine many trees to improve performance. They are often a better choice for structured, tabular data. They can capture nonlinear relationships, require little preprocessing and easy to interpret.

There're two key decisions have to make during the algorithm:

1. How to choose feature to split
2. When to stop

In this project, we will explore tree-based models, understand how they work and build examples to see how they are trained and used.

## Entropy, Impurity and Information Gain

To choose which feature to split, we need the metric `entropy`. Entropy measures how pure or mixed a set of samples is: when a node contains many different labels, its entropy is high and the data is impure; when most samples belong to the same class, entropy is low and the node is relatively pure. In a decision tree, we want to split the data in a way that reduces this uncertainty as much as possible, which is why the feature with the largest information gain is usually chosen.

For binary classification, the entropy of a node is:

$$H(D) = -p_1 log_2(p_1) - p_0 log_2(p_0)$$

where $p_1$ and $p_0$ are the proportions of positive and negative samples. After a split, the information gain is:

$$IG = H(p_{root}) - \left(w_{left} H(p_{left}) + w_{right} H(p_{right})\right)$$

Here, $H(p_{root})$ is the entropy of the parent node, $H(p_{left})$ and $H(p_{right})$ are the entropies of the left and right child nodes, and $w_{left}$, $w_{right}$ are the proportions of samples in each child split. This tells us how much uncertainty is reduced by the split.

### Code Example

We build a decision tree on a sample animals dataset to predict whether it's a cat or not. The dataset is as follows:

| Ear Shape |  Face Shape |  Whiskers |   Cat  |
|:---------:|:-----------:|:---------:|:------:|
|  Pointy   |   Round     |  Present  |    1   |
|  Floppy   |  Not Round  |  Present  |    1   |
|  Floppy   |  Round      |  Absent   |    0   |
|  Pointy   |  Not Round  |  Present  |    0   |
|  Pointy   |   Round     |  Present  |    1   |
|  Pointy   |   Round     |  Absent   |    1   |
|  Floppy   |  Not Round  |  Absent   |    0   |
|  Pointy   |   Round     |  Absent   |    1   |
|  Floppy   |   Round     |  Absent   |    0   |
|  Floppy   |   Round     |  Absent   |    0   |

Starting from the whole data as the root node, we caculate the entropy and information gain with each feature, then decide split on which feature. The process continues recursively until meet the criteria. Implementation codes refer to [decision_tree.py](./decision_tree.py)

## Tree Ensembles

A single decision tree can overfit its training data and be sensitive to small changes in that data. Tree ensembles combine predictions from multiple trees, which can improve stability and generalization.

### Bagging

Bagging trains each tree on little different samples drawn from the training data. The trees predictions are then combined by majority vote for classification or averaging for regression. A random forest adds features randomly as subset of features when choosing each split.

### Boosting


