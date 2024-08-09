import numpy as np
from scipy.spatial import distance_matrix

class LOF:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X):
        self.X = X
        self.dists = distance_matrix(X, X)

    def _local_reachability_density(self, x):
        k_neighbors = np.argsort(self.dists[x])[:self.k + 1]
        reach_dists = np.maximum(self.dists[x][k_neighbors], self.dists[k_neighbors][:, x])
        lrd = len(k_neighbors) / np.sum(reach_dists)
        return lrd

    def predict(self, X):
        lrd_scores = np.array([self._local_reachability_density(x) for x in range(len(X))])
        return lrd_scores < np.percentile(lrd_scores, 5)

# Example
X = np.array([[1], [2], [2.5], [3], [3.5], [5], [10]])  # 10 is an anomaly
model = LOF(k=2)
model.fit(X)
print("Anomalies detected:", model.predict(X))
