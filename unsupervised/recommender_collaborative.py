import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

def normalizeRatings(Y, R):
    """
    Preprocess data by subtracting mean rating for every movie (every row).
    Only include real ratings R(i,j)=1.
    [Ynorm, Ymean] = normalizeRatings(Y, R) normalized Y so that each movie
    has a rating of 0 on average. Unrated moves then have a mean rating (0)
    Returns the mean rating in Ymean.
    """
    Ymean = (np.sum(Y*R,axis=1)/(np.sum(R, axis=1)+1e-12)).reshape(-1,1)
    Ynorm = Y - np.multiply(Ymean, R) 
    return(Ynorm, Ymean)

def load_small_data():
    """
    Load model parameters, ratings, and movie metadata from the resources folder

    Returns:
        X, W, b, num_movies, num_features, num_users, Y, R, movie_list, movie_df
    """
    resources = Path(__file__).resolve().parent / "resources"
    X = pd.read_csv(resources / "small_movies_X.csv", header=None).to_numpy()
    W = pd.read_csv(resources / "small_movies_W.csv", header=None).to_numpy()
    b = pd.read_csv(resources / "small_movies_b.csv", header=None).to_numpy().reshape(1, -1)
    Y = pd.read_csv(resources / "small_movies_Y.csv", header=None).to_numpy()
    R = pd.read_csv(resources / "small_movies_R.csv", header=None).to_numpy()

    num_movies, num_features = X.shape
    num_users = W.shape[0]

    movie_df = pd.read_csv(resources / "small_movie_list.csv", index_col=0)
    movie_list = movie_df["title"].to_list()
    return (X, W, b, num_movies, num_features, num_users, Y, R, movie_list, movie_df)

def cost_func(X, W, b, Y, R, regular_lambda):
    """
    Calculate the cost.
      regular_lambda (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    lost = (np.matmul(X, W.T) + b - Y) * R
    cost = 0.5 * np.sum(lost**2)
    cost += (regular_lambda / 2) * (np.sum(X**2) + np.sum(W**2))
    return cost

def test_case_1(cost_func, X, W, b, Y, R):
    num_users_r = 4
    num_movies_r = 5 
    num_features_r = 3
    X_r = X[:num_movies_r, :num_features_r]
    W_r = W[:num_users_r,  :num_features_r]
    b_r = b[0, :num_users_r].reshape(1,-1)
    Y_r = Y[:num_movies_r, :num_users_r]
    R_r = R[:num_movies_r, :num_users_r]

    # Evaluate cost function
    J = cost_func(X_r, W_r, b_r, Y_r, R_r, 0);
    print(f"Cost: {J:0.2f}")

    # Evaluate cost function with regularization 
    J = cost_func(X_r, W_r, b_r, Y_r, R_r, 1.5);
    print(f"Cost (with regularization): {J:0.2f}")




def create_my_ratings(num_movies):
    # Initialize ratings
    my_ratings = np.zeros(num_movies)

    my_ratings[2700] = 5 
    my_ratings[2609] = 2
    my_ratings[929]  = 5
    my_ratings[246]  = 5
    my_ratings[2716] = 3
    my_ratings[1150] = 5
    my_ratings[382]  = 2
    my_ratings[366]  = 5
    my_ratings[622]  = 5
    my_ratings[988]  = 3
    my_ratings[2925] = 1
    my_ratings[2937] = 1
    my_ratings[793]  = 5
    # my_rated = [i for i in range(len(my_ratings)) if my_ratings[i] > 0]

    # print('\nMy ratings:\n')
    # for i in range(len(my_ratings)):
    #     if my_ratings[i] > 0 :
    #         print(f'Rated {my_ratings[i]} for  {movie_df.loc[i,"title"]}')
    return my_ratings

if __name__ == "__main__":

    # load data
    X, W, b, num_movies, num_features, num_users, Y, R, movie_list, movie_df = load_small_data()

    print(f"X shape: {X.shape}")
    print(f"W shape: {W.shape}")
    print(f"b shape: {b.shape}")
    print(f"Y shape: {Y.shape}")
    print(f"Total users: {num_users}")
    print(f"Total movies: {num_movies}")
    print(f"Movie features: {num_features}\n")


    # test_case_1(cost_func, X, W, b, Y, R)

    # add my ratings
    my_ratings = create_my_ratings(num_movies)
    Y = np.c_[my_ratings, Y]
    R = np.c_[(my_ratings != 0).astype(int), R]

    # Normalize the Dataset
    Ynorm, Ymean = normalizeRatings(Y, R)

