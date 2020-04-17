from cart import CART
import numpy as np
from sklearn.utils import resample


class RandomForest:
    def __init__(self, n_trees, s_bootstrap, n_features, max_depth=10, min_size=6):
        self.n_trees = n_trees
        self.s_bootstrap = s_bootstrap
        self.n_features = n_features
        self.forest = []
        self.max_depth = max_depth
        self.min_size = min_size

    def fit(self, X, y):
        self.forest = []
        for t in range(self.n_trees):
            X_b, y_b = resample(X, y, n_samples=self.s_bootstrap, replace=False)
            tree = CART(max_depth=self.max_depth, min_size=self.min_size, subsample_size=self.n_features)
            tree.fit(X_b, y_b)
            self.forest.append(tree)


class DecisionForest:
    # Each tree uses a random subsample of features
    def __init__(self, n_trees, n_features, max_depth=10, min_size=6):
        self.n_trees = n_trees
        self.n_features = n_features
        self.forest = []
        self.max_depth = max_depth
        self.min_size = min_size

    def fit(self, X, y):
        self.forest = []
        for t in range(self.n_trees):
            f_idx = np.random.choice(range(X.shape[1] - 1), size=self.n_features, replace=False)
            tree = CART(max_depth=self.max_depth, min_size=self.min_size)
            tree.fit(X[:, f_idx], y)
            self.forest.append(tree)