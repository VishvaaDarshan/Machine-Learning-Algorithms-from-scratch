import numpy as np 
from collections import Counter

class KneargestNeighbouts:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    def euclidean_distance(self, a, b):
        return np.sqrt(np.sum((a-b)**2))
    
    
    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]  
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions
    
    def predict2(self, X):
        predictions = []
        for x in X:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]  
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])
        return predictions
    
# Example usage

X_train  = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
y_train  = np.array([0, 0, 1, 1, 0, 1])
model = KneargestNeighbouts(k=3)
model.fit(X_train, y_train)
print("Predicted value for [3, 4]:", model.predict(np.array([[3, 4]])))
