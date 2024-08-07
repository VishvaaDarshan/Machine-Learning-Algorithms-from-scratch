import numpy as np

class LinearRegression:
    def __init__(self):
        self.b0 = 0
        self.b1 = 0
    
    def fit(self, X, y):
        X_mean = np.mean(X)
        y_mean = np.mean(y)
        numerator = np.sum((X - X_mean) * (y - y_mean))
        denominator = np.sum((X - X_mean) ** 2)
        self.b1 = numerator / denominator
        self.b0 = y_mean - self.b1 * X_mean
        
    def predict(self, X):
        return self.b0 + self.b1 * X
    
# Example of the above algorithm 

X = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 3, 2, 5])
model = LinearRegression()
model.fit(X, y)
print("Predicted value for X = 6 is", model.predict(6))