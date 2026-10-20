from __future__ import annotations

from dataclasses import dataclass
from math import log2
from typing import Optional, Sequence


MAX_DEPTH = 5
IG_THRESHOLD = 0.2
NUM_THRESHOLD = 2

@dataclass
class Node:
    is_leaf: bool
    depth: int
    split_feature_index: Optional[int]
    predict_class: int
    left_node: Optional[Node] = None
    right_node: Optional[Node] = None


class CatDecisionTree:

    def __init__(self, max_depth: int = MAX_DEPTH, 
                 information_gain_threshold: float = IG_THRESHOLD, 
                 min_samples_split: int = NUM_THRESHOLD) -> None:
        self.max_depth = max_depth
        self.information_gain_threshold = information_gain_threshold
        self.min_samples_split = min_samples_split
        self.root: Optional[Node] = None
        self.feature_count = 0

    def train(self, features, labels):

        if len(features) != len(labels):
            raise ValueError("features and labels must contain the same number of samples.")

        self.feature_count = len(features[0])

        dataset: list[tuple[Sequence[int], int]] = []
        for feature_row, label in zip(features, labels):
            dataset.append((feature_row, int(label)))

        self.root = self._build_tree(dataset, tuple(range(self.feature_count)), 0)

    def _build_tree(self, dataset, available_features, depth) -> Node:
        labels = [label for _, label in dataset]
        majority_class = self._majority_class(labels)
        is_pure = all(label == labels[0] for label in labels)

        if (is_pure or depth >= self.max_depth or len(dataset) < self.min_samples_split or not available_features):
            return Node(True, depth, None, majority_class)

        split_feature_index, information_gain = self._best_split(dataset, available_features)

        if (split_feature_index == -1 or information_gain < self.information_gain_threshold):
            return Node(True, depth, None, majority_class)

        left_dataset = [sample for sample in dataset if sample[0][split_feature_index] == 1]
        right_dataset = [sample for sample in dataset if sample[0][split_feature_index] == 0]
        # do not reuse the feature used for splitting
        remaining_features = tuple(index for index in available_features if index != split_feature_index)

        # recursively splitting
        left_node = self._build_tree(left_dataset, remaining_features, depth + 1)
        right_node = self._build_tree(right_dataset, remaining_features, depth + 1)
        
        return Node(is_leaf=False, depth=depth, split_feature_index=split_feature_index, predict_class=majority_class, left_node=left_node, right_node=right_node)

    @staticmethod
    def _majority_class(labels: Sequence[int]) -> int:
        positive_count = sum(labels)
        negative_count = len(labels) - positive_count
        return 1 if positive_count > negative_count else 0

    @staticmethod
    def _cal_entropy(labels: Sequence[int]) -> float:
        if not labels:
            return 0.0

        positive_proportion = sum(labels) / len(labels)
        if positive_proportion == 0.0 or positive_proportion == 1.0:
            return 0.0

        negative_proportion = 1.0 - positive_proportion
        return -(positive_proportion * log2(positive_proportion) + negative_proportion * log2(negative_proportion))

    def _best_split(self, dataset, feature_indices) -> tuple[int, float]:
        '''calculate the information gain and choose the best split'''

        parent_entropy = self._cal_entropy([label for _, label in dataset])
        best_feature_index: int = -1
        best_information_gain = 0.0

        for feature_index in feature_indices:
            left_labels = [label for feature, label in dataset if feature[feature_index] == 1]
            right_labels = [label for feature, label in dataset if feature[feature_index] == 0]
            if not left_labels or not right_labels:
                continue

            left_weight = len(left_labels) / len(dataset)
            right_weight = len(right_labels) / len(dataset)
            information_gain = parent_entropy - (
                left_weight * self._cal_entropy(left_labels)
                + right_weight * self._cal_entropy(right_labels)
            )
            if information_gain > best_information_gain:
                best_information_gain = information_gain
                best_feature_index = feature_index

        return best_feature_index, best_information_gain

    def predict(self, features: Sequence[int]) -> int:
        if self.root is None:
            raise RuntimeError("Train the decision tree before predict.")
        if len(features) != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, received {len(features)}."
            )
        if any(value not in (0, 1) for value in features):
            raise ValueError("Features must be encoded as 0 or 1.")

        node = self.root
        while not node.is_leaf:
            feature_index = node.split_feature_index
            if feature_index is None:
                raise RuntimeError("Node splitting feature error!")
            child = node.left_node if features[feature_index] == 1 else node.right_node
            if child is None:
                raise RuntimeError("Child node error!")
            node = child

        return node.predict_class

    def predict_many(self, features: Sequence[Sequence[int]]) -> list[int]:
        return [self.predict(row) for row in features]


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

    tree = CatDecisionTree()
    tree.train(features, labels)
    predictions = tree.predict_many(features)
    print(f"Predictions: {predictions}")
    print(f"Training accuracy: {sum(p == y for p, y in zip(predictions, labels)) / len(labels):.0%}")
