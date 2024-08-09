import numpy as np

class SVM:
    def __init__(self, lr=0.001, epochs = 1000, lambda_param=0.01):
        self.lr = lr
        self.epochs = epochs
        self.lambda_param = lambda_param
        self.w = None
        self.b = None
          
    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.b = 0
        for _ in range(self.epochs):
            for i, x_i in enumerate(X):
                if y[i] * (np.dot(x_i, self.w) - self.b) < 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[i]))
                    self.b -= self.lr * y[i]
            for i, x_i in enumerate(X):
                if y[i] * (np.dot(x_i, self.w) - self.b) >= 1:
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y[i]))
                    self.b -= self.lr * y[i]
                else:
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) - self.b)
    
# Example usage
X = np.array([[1, 2], [2, 3], [3, 4], [6, 7], [7, 8], [8, 9]])
y = np.array([-1, -1, -1, 1, 1, 1])
model = SVM()
model.fit(X, y)
print("Predicted value for [5, 5]:", model.predict(np.array([[5, 5]])))