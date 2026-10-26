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

class CollaborativeFiltering(nn.Module):

    def __init__(self, X, W, b):
        super().__init__()
        num_movies, num_features = X.shape
        num_users = W.shape[0]
        if W.shape[1] != num_features or b.size != num_users:
            raise ValueError("X, W, and b have incompatible shapes.")

        # initialize embeddings and bias
        self.movie_embeddings = nn.Embedding(num_movies, num_features)
        self.user_embeddings = nn.Embedding(num_users, num_features)
        self.user_bias = nn.Embedding(num_users, 1)

        # assign values to each
        with torch.no_grad():
            self.movie_embeddings.weight.copy_(torch.as_tensor(X, dtype=torch.float32))
            self.user_embeddings.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
            self.user_bias.weight.copy_(torch.as_tensor(b, dtype=torch.float32).reshape(num_users, 1))

    def forward(self):
        movie_ids = torch.arange(
            self.movie_embeddings.num_embeddings,
            device=self.movie_embeddings.weight.device
        )
        user_ids = torch.arange(
            self.user_embeddings.num_embeddings,
            device=self.user_embeddings.weight.device
        )
        movie_vectors = self.movie_embeddings(movie_ids)
        user_vectors = self.user_embeddings(user_ids)
        user_bias = self.user_bias(user_ids).T
        return movie_vectors @ user_vectors.T + user_bias


def train_collaborative_filtering(X, W, b, Y, R, regular_lambda=1.0, learning_rate=1e-2, epochs=100):
    """Optimize the initialized movie/user embeddings and user biases"""
    if regular_lambda < 0:
        raise ValueError("regular_lambda must be non-negative.")
    if learning_rate <= 0 or epochs <= 0:
        raise ValueError("learning_rate and epochs must be positive.")
    if Y.shape != R.shape or Y.shape != (X.shape[0], W.shape[0]):
        raise ValueError("Y and R must match the movie and user dimensions.")
    if not np.any(R):
        raise ValueError("Cannot train without any observed ratings.")

    model = CollaborativeFiltering(X, W, b)
    ratings = torch.as_tensor(Y, dtype=torch.float32)
    observed = torch.as_tensor(R, dtype=torch.float32)
    num_observed = observed.sum()
    # do not use `AdamW` because we manually add `regular_lambda` in loss calculation
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model()
        error = (predictions - ratings) * observed
        loss = (0.5 / num_observed) * torch.sum(error**2)
        # Scale regularization too, preserving its strength relative to the summed objective.
        loss = loss + (regular_lambda / (2 * num_observed)) * (
            torch.sum(model.movie_embeddings.weight**2)
            + torch.sum(model.user_embeddings.weight**2)
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}   Average Loss:{loss.item():.3f}")

    return model, losses


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

    # The added user's embedding and bias start at zero and are learned with
    # the existing movie/user parameters.
    W = np.vstack((W, np.zeros((1, num_features), dtype=W.dtype)))
    b = np.hstack((b, np.zeros((1, 1), dtype=b.dtype)))

    Ynorm, Ymean = normalizeRatings(Y, R)
    model, losses = train_collaborative_filtering(X,W,b,Ynorm,R,regular_lambda=1.0,learning_rate=1e-2,epochs=100)

    # print(f"Initial cost: {losses[0]:.2f}")
    # print(f"Final cost: {losses[-1]:.2f}")

    with torch.no_grad():
        normalized_predictions = model().cpu().numpy()

    # add back mean score
    new_user_id = 0
    predictions = normalized_predictions[:, new_user_id] + Ymean[:, 0]

    rated_movies = np.flatnonzero(my_ratings > 0)
    rating_comparison = pd.DataFrame(
        {
            "Movie": [movie_list[movie_id] for movie_id in rated_movies],
            "Rated": my_ratings[rated_movies],
            "Predicted": predictions[rated_movies],
        }
    )
    rating_comparison["Difference"] = (
        rating_comparison["Predicted"] - rating_comparison["Rated"]
    )

    print("\nRatings compared with the predictions:")
    print(
        rating_comparison.to_string(
            index=False,
            formatters={
                "Rated": "{:.2f}".format,
                "Predicted": "{:.2f}".format,
                "Difference": "{:+.2f}".format,
            }
        )
    )

    # only recommend unrated movies
    unrated_movies = np.flatnonzero(my_ratings == 0)
    recommended_movies = unrated_movies[
        np.argsort(predictions[unrated_movies])[-10:][::-1]
    ]

    print("\nTop recommendations:")
    for movie_id in recommended_movies:
        print(f"{predictions[movie_id]:.2f} - {movie_list[movie_id]}")

        
