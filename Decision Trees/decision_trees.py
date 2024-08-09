import numpy as np
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

class DecisionTree:
    def gini(self, y):
        m = len(y)
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))

    def split(self, X, y, feature, threshold):
        left_idx = np.where(X[:, feature] <= threshold)
        right_idx = np.where(X[:, feature] > threshold)
        return X[left_idx], X[right_idx], y[left_idx], y[right_idx]

    def best_split(self, X, y):
        best_gini = float("inf")
        best_split = None
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                X_left, X_right, y_left, y_right = self.split(X, y, feature, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                gini_split = (len(y_left) / len(y)) * self.gini(y_left) + (len(y_right) / len(y)) * self.gini(y_right)
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_split = {
                        "feature": feature,
                        "threshold": threshold,
                        "left": (X_left, y_left),
                        "right": (X_right, y_right),
                    }
        return best_split

    def build_tree(self, X, y, depth=0, max_depth=3):
        if depth == max_depth or len(set(y)) == 1:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)
        
        split_info = self.best_split(X, y)
        if not split_info:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)