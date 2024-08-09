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
        
        left = self.build_tree(*split_info["left"], depth + 1, max_depth)
        right = self.build_tree(*split_info["right"], depth + 1, max_depth)
        return Node(feature=split_info["feature"], threshold=split_info["threshold"], left=left, right=right)

    def predict_tree(self, node, X):
        if node.value is not None:
            return node.value
        if X[node.feature] <= node.threshold:
            return self.predict_tree(node.left, X)
        else:
            return self.predict_tree(node.right, X)

    def fit(self, X, y):
        self.tree = self.build_tree(X, y)

    def predict(self, X):
        return [self.predict_tree(self.tree, x) for x in X]

# Example usage
X = np.array([[2, 3], [1, 1], [3, 2], [7, 8], [8, 8], [9, 9]])
y = np.array([0, 0, 0, 1, 1, 1])
model = DecisionTree()
model.fit(X, y)
print("Predicted value for [5, 5]:", model.predict(np.array([[5, 5]])))