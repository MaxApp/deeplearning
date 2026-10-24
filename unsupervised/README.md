# Unsupervised Learning

Unsupervised learning helps discover patterns and structure in data without labeled examples. Common approaches include `clustering` similar data points, `reducing dimensions` to reveal important features, and `detecting unusual observations`. It is useful for exploring datasets and uncovering groups or relationships that were not specified in advance.

## Clustering

Clustering is an unsupervised learning technique that groups data points based on their similarity without requiring predefined labels. A common method is `K-means` which partitions data into a chosen number of clusters through two repeating steps:

1. **Assign points:** Assign each data point to its nearest cluster center (centroid).
2. **Update centers:** Recalculate each centroid as the mean of the points assigned to that cluster.

Repeat these steps until the assignments or centroids stop changing. Clustering can reveal patterns such as customer groups or similar documents.

The results of [k-means sample](./k_means.py) shows below:

![k-means](./imgs/k-means.png)

## Anomaly Detection

Anomaly detection identifies observations that differ substantially from the patterns in the majority of the data. It is useful for finding unusual transactions, equipment failures, or suspicious network activity, especially when labeled examples of anomalies are scarce.

One approach models normal data with a `Gaussian distribution`. After estimating the mean and variance of each feature from normal examples, the model learns the distribution of the normal data, observations with low probability are flagged as anomalies. This approach works best when normal data is reasonably well described by the chosen distribution.

Estimate the Gaussian parameters as:

$$
\mu_j = \frac{1}{m}\sum_{i=1}^{m} x_j^{(i)}, \qquad
\sigma_j^2 = \frac{1}{m}\sum_{i=1}^{m}\left(x_j^{(i)}-\mu_j\right)^2
$$

Feature scaling and a suitable validation strategy are important when comparing methods and choosing a threshold. We need to select a threshold based on a cross validation set with labeled data by `F1-score`.

![anomalous points](./imgs/anomalous.png)

Other methods make different assumptions about what makes an observation unusual.

## Dimension Reduction
