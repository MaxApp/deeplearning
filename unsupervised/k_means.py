import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt


def k_means(points, k, max_iterations=100, n_init=20, random_state=None):
    
    if k < 1 or k > len(points):
        raise ValueError("k must be between 1 and the number of points")

    # new edition of random generator
    rng = np.random.default_rng(random_state)
    best_total_distance = np.inf
    best_indicies = None
    best_centroids = None

    # initial multiple times
    for _ in range(n_init):

        # random select k points as initial centroids each time
        initial_indices = rng.choice(len(points), size=k, replace=False)
        centroids = points[initial_indices].copy()

        for _ in range(max_iterations):
            indicies = find_best_centroids(points, centroids)
            updated_centroids = calculate_new_centroids(points, indicies, centroids, k)

            if np.allclose(centroids, updated_centroids):
                centroids = updated_centroids
                break
            centroids = updated_centroids

        indicies = find_best_centroids(points, centroids)
        total_distance = np.sum((points - centroids[indicies]) ** 2)
        if total_distance < best_total_distance:
            best_total_distance = total_distance
            best_indicies = indicies.copy()
            best_centroids = centroids.copy()

    assert best_indicies is not None and best_centroids is not None
    return best_indicies, best_centroids

def find_best_centroids(points, centroids):
    """
    assign each point to the closest centroids
    Args:
        points
        centroids
    Return:
        centroid indicies
    """

    indicies = np.zeros(points.shape[0], dtype=int)
    for i, p in enumerate(points):
        distances = []
        for c in centroids:
            d_ij = np.linalg.norm(p-c)
            distances.append(d_ij)
        indicies[i] = np.argmin(distances)

    return indicies

def calculate_new_centroids(points, indicies, old_centroids, k):
    updated_centroids = old_centroids.copy()
    for i in range(k):
        belonged_points = points[indicies == i]
        if len(belonged_points) > 0:
            updated_centroids[i] = np.mean(belonged_points, axis=0)
    return updated_centroids


if __name__ == "__main__":

    current_directory = Path(__file__).resolve().parent
    sample_points = np.load(current_directory / "k_means_dataset.npy")

    sample_rng = np.random.default_rng(12)
    k = 3
    cluster_labels, centroids = k_means(
        sample_points, k, n_init=20, random_state=7
    )

    for cluster_index in range(k):
        cluster_points = sample_points[cluster_labels == cluster_index]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            label=f"Cluster {cluster_index + 1}",
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        color="black",
        marker="X",
        s=150,
        label="Centroids",
    )
    plt.title("K-means Clustering (K=3)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.show()