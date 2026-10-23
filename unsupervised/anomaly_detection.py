import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def estimate_gaussian(X): 
    """
    calculates mean and variance of all features 
    Args:
        X (ndarray): (m, n) Data matrix
    Returns:
        mu (ndarray): (n,) Mean of all features
        var (ndarray): (n,) Variance of all features
    """
    m, n = X.shape
    mu = 1 / m * np.sum(X, axis = 0)
    var = 1 / m * np.sum((X - mu) ** 2, axis = 0)
    return mu, var














if __name__ == "__main__":

    current_directory = Path(__file__).resolve().parent

    # load sample data
    X_train = np.load(current_directory / "resources/X_anomaly_detection.npy")
    X_val = np.load(current_directory / "resources/X_val_anomaly_detection.npy")
    y_val = np.load(current_directory / "resources/y_val_anomaly_detection.npy")

    print("The first 5 elements of X_train are:\n", X_train[:5])  
    print("The first 5 elements of X_val are:\n", X_val[:5])  
    print("The first 5 elements of y_val are:\n", y_val[:5])  

    # plt.scatter(X_train[:, 0], X_train[:, 1], marker='x', c='b') 

    # # Set the title
    # plt.title("The dataset")
    # # Set axis range
    # plt.axis((0., 30., 0., 30.))
    # plt.show()

    mu, var = estimate_gaussian(X_train)              

    print("Mean of each feature:", mu)
    print("Variance of each feature:", var)