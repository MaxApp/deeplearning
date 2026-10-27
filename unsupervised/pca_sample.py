import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


if __name__ == "__main__":

    X = np.array([[ 99,  -1],
       [ 98,  -1],
       [ 97,  -2],
       [101,   1],
       [102,   1],
       [103,   2]])

    pca = PCA(n_components=1)
    pca.fit(X)
    print(f"explained_variance_ratio: {pca.explained_variance_ratio_}")

    X_trans = pca.transform(X)
    X_reduced = pca.inverse_transform(X_trans)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(X[:, 0], X[:, 1])
    axes[0].set_title("Before PCA")

    axes[1].scatter(X_reduced[:, 0], X_reduced[:, 1])
    axes[1].set_title("After PCA (1 component)")

    for ax in axes:
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    plt.show()

    