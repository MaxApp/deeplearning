# import pandas as pd
import numpy as np

class CatDecisionTree:

    def __init__(self) -> None:
        self.tree = []
        pass

    def build_tree(self):
        pass

    def _cal_entropy(self):
        pass

    def _information_gain(self):
        pass

    def _best_split(self):
        pass

    def train(self):
        pass

    def predict(self):
        pass


def binary_entropy(labels):
    cat_count = np.sum(labels == 1)
    if cat_count == 0 or cat_count == labels.size:
        return -np.log2(labels.size)
    else:
        p1 = cat_count/labels.size
        return -p1*np.log2(p1)-(1-p1)*np.log2(1-p1)

def information_gain(root_labels, left_labels, right_labels):
    pass

def split_by_feature():

    pass

def build_tree():

    pass

if __name__ == "__main__":

    """
    The training dataset represented as:
    - Ear Shape: Pointy = 1, Floppy = 0
    - Face Shape: Round = 1, Not Round = 0
    - Whiskers: Present = 1, Absent = 0
    """
    train_dataset = np.array([[1, 1, 1],
                            [0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 1],
                            [1, 1, 1],
                            [1, 1, 0],
                            [0, 0, 0],
                            [1, 1, 0],
                            [0, 1, 0],
                            [0, 1, 0]])

    labels = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])

    