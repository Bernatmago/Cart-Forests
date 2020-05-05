from cart import CART
import numpy as np
from sklearn.utils import resample



class RandomForest:
    def __init__(self, n_trees, s_bootstrap, n_features, max_depth=-1, min_size=1):
        self.n_trees = n_trees
        self.s_bootstrap = s_bootstrap
        self.n_features = n_features
        self.forest = []
        self.max_depth = max_depth
        self.min_size = min_size
        self.rule_counts = {}

    def fit(self, X, y, numerical_idx=None):
        if numerical_idx is None:
            numerical_idx = []
        self.forest = []
        for t in range(self.n_trees):
            X_b, y_b = resample(X, y, n_samples=self.s_bootstrap, replace=False)
            tree = CART(max_depth=self.max_depth, min_size=self.min_size, subsample_size=self.n_features)
            tree.fit(X_b, y_b, numerical_idx)
            # print('RF ', t)
            self.forest.append(tree)

    def predict(self, X):
        votes = np.stack([t.predict(X) for t in self.forest], axis=1)
        preds = []
        for sample in votes:
            (v, c) = np.unique(sample, return_counts=True)
            i = np.argmax(c)
            preds.append(v[i])
        return np.array(preds)

    def rule_count(self):
        for t in self.forest:
            r = t.rule_count()
            keys = list(r.keys())
            for f in keys:
                f_v = self.rule_counts.get(f, {})
                r_v = r.get(f, {})
                self.rule_counts[f] = {k: f_v.get(k, 0) + r_v.get(k, 0) for k in set(f_v) | set(r_v)}
        return self.rule_counts


class DecisionForest:
    # Each tree uses a random subsample of features
    def __init__(self, n_trees, n_features, max_depth=-1, min_size=1):
        self.n_trees = n_trees
        self.n_features = n_features
        self.forest = []
        self.tree_feats = []
        self.max_depth = max_depth
        self.min_size = min_size
        self.rule_counts = {}

    def fit(self, X, y, numerical_idx=None):
        if numerical_idx is None:
            numerical_idx = []
        self.forest = []
        self.tree_feats = []
        n_feats = self.n_features
        for t in range(self.n_trees):
            if isinstance(self.n_features, str):
                n_feats = int(np.random.uniform(1, X.shape[1]))
            f_idx = np.sort(np.random.choice(range(X.shape[1] - 1), size=n_feats, replace=False))
            tree = CART(max_depth=self.max_depth, min_size=self.min_size)
            n_idx = [x for x in range(len(f_idx)) if f_idx[x] in numerical_idx]
            tree.fit(X[:, f_idx], y, n_idx)
            # print('DF ', t)
            self.forest.append(tree)
            self.tree_feats.append(f_idx)

    def predict(self, X):
        votes = np.stack([t.predict(X[:, self.tree_feats[c]]) for c, t in enumerate(self.forest)], axis=1)
        preds = []
        for sample in votes:
            (v, c) = np.unique(sample, return_counts=True)
            i = np.argmax(c)
            preds.append(v[i])
        return np.array(preds)

    def rule_count(self):
        for i, t in enumerate(self.forest):
            r = t.rule_count()
            keys = list(r.keys())
            keys.sort()
            real_keys = self.tree_feats[i]

            for k, rk in zip(keys, real_keys):
                f_v = self.rule_counts.get(rk, {})
                r_v = r.get(k, {})
                self.rule_counts[rk] = {k: f_v.get(k, 0) + r_v.get(k, 0) for k in set(f_v) | set(r_v)}
        return self.rule_counts
