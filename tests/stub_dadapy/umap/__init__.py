import numpy as np
class UMAP:
    def __init__(self, n_components=2, n_neighbors=15, random_state=0):
        self.k, self.n_neighbors = n_components, n_neighbors
    def fit_transform(self, X):
        X = X - X.mean(0); return X @ np.linalg.svd(X, full_matrices=False)[2][:self.k].T
