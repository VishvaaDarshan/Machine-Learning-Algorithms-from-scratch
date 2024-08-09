import numpy as np 
from sklearn.utils import resample

class IsolationForest:
    def __init__(self, n_trees = 100, max_samples = 'auto'):
        self.n_trees = n_trees
        self.max_samples = max_samples
    
    def fit(self, X):
        self.trees = [self._build_tree(X) for _ in range(self.n_trees)]
        
    def _build_tree(self, X):
        n_samples, n_feature = X.shape
        if self.max_samples == 'auto':
            self.max_samples = min(256, n_samples)
        else:
            self.max_samples = self.max_samples
            
        X_sample = resample(X, n_samples=self.max_samples, random_state=0)
        return self._fit_tree(X_sample)
    
    def _fit_tree(self, X):
        if len(X) <=1:
            return None
        
        n_samples, n_features = X.shape
        feature = np.random.randint(n_features)
        threshold = np.random.uniform(np.min(X[:, feature]), np.max(X[:, feature]))
        
        left = X[X[:, feature] < threshold]
        right = X[X[:, feature] > threshold]
        
        return(feature, threshold, self._fit_tree(left), self._fit_tree(right))
    
    def _score_samples(self, X, tree):
        if tree is None:
            return np.zeros(X.shape[0])
        
        
        feature, threshold, left, right = tree
        left_indices = X[:, feature] <= threshold
        right_indices = X[:, feature] > threshold
        
        scores = np.zeros(X.shape[0])
        scores[left_indices] = self._score_samples(X[left_indices], left)
        scores[right_indices] = self._score_samples(X[right_indices], right)
        scores = scores + 1
        return scores
    
    def predict(self, X):
        scores = np.zeros(X.shape[0])
        for tree in self.trees:
            scores += self._score_samples(X, tree) 
        scores = scores / self.n_trees
        return scores > np.percentile(scores, 95)
    
#Example
X = np.array([[1], [2], [2.5], [3], [3.5], [5], [10]])  # 10 is an anomaly
model = IsolationForest(n_trees=10)
model.fit(X)
print("Anomalies detected:", model.predict(X))