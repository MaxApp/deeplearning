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


def calculate_gaussian(X, mu, var):
    """
    Computes the probability density.
    If var is a matrix, it is treated as the covariance matrix.
    If var is a vector, it is treated as the var values of the variances in each dimension
    """
    
    k = len(mu)
    
    if var.ndim == 1:
        var = np.diag(var)
        
    X = X - mu
    p = (2* np.pi)**(-k/2) * np.linalg.det(var)**(-0.5) * \
        np.exp(-0.5 * np.sum(np.matmul(X, np.linalg.pinv(var)) * X, axis=1))
    
    return p


def select_threshold(y_val, p_val): 
    """
    Finds the best threshold by validation set
    
    Args:
        y_val (ndarray): labels
        p_val (ndarray): probability density
        
    Returns:
        epsilon (float): threshold 
        F1 (float): F1 score
    """ 

    best_epsilon = 0
    best_f1 = 0
    f1_score = 0
    
    step_size = (max(p_val) - min(p_val)) / 1000
    
    for epsilon in np.arange(min(p_val), max(p_val), step_size):
    
        predictions = (p_val < epsilon)
        tp = np.sum((predictions == 1) & (y_val == 1))
        fn = np.sum((predictions == 0) & (y_val == 1))
        fp = sum((predictions == 1) & (y_val == 0))
        prec = tp / (tp + fp) if tp + fp > 0 else 0
        rec = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0
        
        if f1_score > best_f1:
            best_f1 = f1_score
            best_epsilon = epsilon
        
    return best_epsilon, best_f1


def predict_show(points, mu, variance, threshold):
    p = calculate_gaussian(points, mu, variance)
    anomalous_mask = p < threshold

    # show anomalous data
    plt.scatter(points[:, 0], points[:, 1], marker='x', c='b') 
    
    # Set the title
    plt.title("The dataset")
    # Set axis range
    plt.axis((0., 30., 0., 30.))
    plt.plot(points[anomalous_mask, 0], points[anomalous_mask, 1], c='r', marker='o', linestyle="",
             markersize=15, markerfacecolor='none')

    plt.show()


if __name__ == "__main__":

    current_directory = Path(__file__).resolve().parent

    # load sample data
    X_train = np.load(current_directory / "resources/X_anomaly_detection.npy")
    X_val = np.load(current_directory / "resources/X_val_anomaly_detection.npy")
    y_val = np.load(current_directory / "resources/y_val_anomaly_detection.npy")

    # print("The first 5 elements of X_train:\n", X_train[:5])
    # print("The first 5 elements of X_val:\n", X_val[:5])
    # print("The first 5 elements of y_val:\n", y_val[:5])

    mu, var = estimate_gaussian(X_train)

    print("Mean of each feature:", mu)
    print("Variance of each feature:", var)

    p_val = calculate_gaussian(X_val, mu, var)
    epsilon, f1 = select_threshold(y_val, p_val)

    print(f"Threshold: {epsilon}    F1-score:{f1}")

    predict_show(X_train, mu, var, epsilon)
