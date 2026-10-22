import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def k_means(points, k, max_iterations=100, n_init=20, random_state=None):
    
    if k < 1 or k > len(points):
        raise ValueError("k must be between 1 and the number of points")

    # new edition of random generator
    rng = np.random.default_rng(random_state)
    best_inertia = np.inf
    best_labels = None
    best_centroids = None

    for _ in range(n_init):
        # random select k points as initial centroids
        initial_indices = rng.choice(len(points), size=k, replace=False)
        centroids = points[initial_indices].copy()

        for _ in range(max_iterations):
            distances = np.linalg.norm(
                points[:, np.newaxis] - centroids, axis=2
            )
            labels = np.argmin(distances, axis=1)

            updated_centroids = centroids.copy()
            for cluster_index in range(k):
                cluster_points = points[labels == cluster_index]
                if len(cluster_points) > 0:
                    updated_centroids[cluster_index] = cluster_points.mean(axis=0)

            if np.allclose(centroids, updated_centroids):
                centroids = updated_centroids
                break
            centroids = updated_centroids

        distances = np.linalg.norm(points[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        inertia = np.sum((points - centroids[labels]) ** 2)
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    assert best_labels is not None and best_centroids is not None
    return best_labels, best_centroids

def find_best_centroids(points, centroids):
    """
    assign each point to the closest centroids
    Args:
        points
        centroids
    Return:
        centroid indicies
    """

    indicies = []
    for p in points:
        distances = []
        for c in centroids:
            d_ij = np.linalg.norm(p-c)
            distances.append(d_ij)
        idx = np.argmin(distances)
        indicies.append(idx)

    return indicies



if __name__ == "__main__":

    current_directory = Path(__file__).resolve().parent
    sample_points = np.load(current_directory / "k_means_dataset.npy")

    # Select an initial set of centroids (3 Centroids)
    initial_centroids = np.array([[3,3], [6,2], [8,5]])

    # Find closest centroids using initial_centroids
    idx = find_best_centroids(sample_points, initial_centroids)

    # Print closest centroids for the first three elements
    print("First three elements in idx are:", idx[:3])

    # sample_rng = np.random.default_rng(12)
    # sample_points = np.load("k_means_dataset.npy")
    # k = 2
    # cluster_labels, centroids = k_means(
    #     sample_points, k, n_init=20, random_state=7
    # )

    # for cluster_index in range(k):
    #     cluster_points = sample_points[cluster_labels == cluster_index]
    #     plt.scatter(
    #         cluster_points[:, 0],
    #         cluster_points[:, 1],
    #         label=f"Cluster {cluster_index + 1}",
    #     )

    # plt.scatter(
    #     centroids[:, 0],
    #     centroids[:, 1],
    #     color="black",
    #     marker="X",
    #     s=150,
    #     label="Centroids",
    # )
    # plt.title("K-means Clustering (K=2)")
    # plt.xlabel("X coordinate")
    # plt.ylabel("Y coordinate")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.axis("equal")
    # plt.show()