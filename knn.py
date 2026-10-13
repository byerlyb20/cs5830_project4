from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.spatial.distance import cdist
import numpy as np

class KnnClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.classes_ = np.unique(self.y)
        return self

    def predict(self, X):
        dist_matrix = cdist(np.asarray(X), self.X)
        nearest_neighbors = np.argsort(dist_matrix, axis=1)
        knn_lookup = nearest_neighbors[:, :self.k]
        knn_labeled = self.y[knn_lookup]
        return np.mean(knn_labeled, axis=1) >= 0.5

    def score(self, X, y, sample_weight=None):
        correct = self.predict(X) == np.asarray(y)
        return np.average(correct)
