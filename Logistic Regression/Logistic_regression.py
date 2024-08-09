import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.theta = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)
        self.theta = np.zeros(X.shape[1])
        for _ in range(self.iterations):
            z = np.dot(X, self.theta)
            h = self.sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / len(y)
            self.theta -= self.lr * gradient

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return self.sigmoid(np.dot(X, self.theta)) >= 0.5

# Example usage
X = np.array([[0.5], [2.3], [2.9], [3.3], [4.0], [5.5]])
y = np.array([0, 0, 0, 1, 1, 1])
model = LogisticRegression()
model.fit(X, y)
print("Predicted value for [3.5]:", model.predict(np.array([[3.5]])))
print("Predicted value for [4.5]:", model.predict(np.array([[4.5]])))


