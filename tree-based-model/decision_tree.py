# import pandas as pd
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

MAX_DEPTH = 5
IG_THRESHOLD = 0.2
NUM_THRESHOLD = 2

@dataclass
class Node:
    is_leaf: bool
    depth: int
    split_feature_index: int
    predict_class: int
    left_node: Node
    right_node: Node

class CatDecisionTree:

    def __init__(self) -> None:
        self.tree = []
        pass

    def build_tree(self, dataset, feature_size, depth):
        if depth >= MAX_DEPTH:
            return
        left_nodes, right_nodes, best_feature_index = self._best_split(dataset, feature_size)
        self.tree.append(left_nodes)
        self.tree.append(right_nodes)
        self.build_tree(left_nodes, feature_size, depth+1)
        self.build_tree(right_nodes, feature_size, depth+1)

    def _cal_entropy(self, dataset):
        labels = np.array([i["label"] for i in dataset])
        cat_count = np.sum(labels == 1)
        p1 = cat_count / labels.size
        if p1 == 0 or p1 == 1:
            return 0
        else:
            return -p1*np.log2(p1)-(1-p1)*np.log2(1-p1)

    def _information_gain(self, root_entropy, left_nodes, right_nodes):
        total_count = len(left_nodes) + len(right_nodes)
        left_entropy = self._cal_entropy(left_nodes)
        right_entropy = self._cal_entropy(right_nodes)
        left_weight = len(left_nodes) / total_count
        right_weight = len(right_nodes) / total_count
        return root_entropy - (left_weight * left_entropy + right_weight * right_entropy)

    def _best_split(self, dataset, feature_size):
        information_gain = -1
        best_feature_index = -1
        left_nodes = []
        right_nodes = []
        root_entropy = self._cal_entropy(dataset)

        for i in feature_size:
            # calculate information gain with each feature
            l_nodes = []
            r_nodes = []
            for ds in dataset:
                if ds["feature"][i] == 1:
                    l_nodes.append(ds)
                else:
                    r_nodes.append(ds)
            ig = self._information_gain(root_entropy, l_nodes, r_nodes)
            if ig > information_gain:
                information_gain = ig
                left_nodes = l_nodes
                right_nodes = r_nodes
                best_feature_index = i

        return left_nodes, right_nodes, best_feature_index

    def train(self):
        pass

    def predict(self):
        pass


if __name__ == "__main__":

    """
    The training dataset represented as:
    - Ear Shape: Pointy = 1, Floppy = 0
    - Face Shape: Round = 1, Not Round = 0
    - Whiskers: Present = 1, Absent = 0
    """
    features = [[1, 1, 1],
                [0, 0, 1],
                [0, 1, 0],
                [1, 0, 1],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 0]]

    labels = [1, 1, 0, 0, 1, 1, 0, 1, 0, 0]

    dataset = []
    for f,l in zip(features, labels):
        dataset.append({"feature": f, "label": l})
    print(dataset)



    