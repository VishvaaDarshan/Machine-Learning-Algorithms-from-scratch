import numpy as np 

class ZScoreAD:
    def __init__(self, threshold=3.5):
        self.threshold = threshold
        self.mean = 0
        self.std = 0

    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)

    def predict(self, X):
        scores = np.abs(X - self.mean) / self.std
        return scores > self.threshold
    
#Example
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100])
model = ZScoreAD(threshold=2)
model.fit(X)
print("Anomalies:", model.predict(X))