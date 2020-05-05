import numpy as np


class CART:
    def __init__(self, max_depth=-1, min_size=1, subsample_size=-1):
        self.max_depth = max_depth
        self.min_size = min_size
        self.subsample_size = subsample_size
        self.root = {}
        self.numerical_idx = []
        self.rule_counts = {}

    def __str__(self):
        self.__print_tree(self.root)

    def fit(self, X, y, numerical_idx=[]):
        X = np.hstack((X, y.reshape(-1, 1)))
        self.rule_counts = {}
        self.numerical_idx = numerical_idx
        self.root = self.__get_split(X)
        self.__split(self.root, 1)

    def predict(self, X):
        preds = []
        for sample in X:
            preds.append(self.__predict_sample(self.root, sample))
        return np.array(preds)

    def rule_count(self):
        return self.rule_counts

    def __print_tree(self, node, depth=0):
        if isinstance(node, dict):
            if node['idx'] in self.numerical_idx:
                print('depth {} [{} <= {}]'.format(depth, node['idx'], node['val']))
            else:
                print('depth {} [{} =? {}]'.format(depth, node['idx'], node['val']))
                self.__print_tree(node['left'], depth=depth + 1)
                self.__print_tree(node['right'], depth=depth + 1)
        else:
            print('depth {} [{}]'.format(depth, node))

        pass

    def __predict_sample(self, node, sample):
        left = False
        if node['idx'] in self.numerical_idx:
            if sample[node['idx']] <= node['val']:
                left = True
        else:
            if sample[node['idx']] == node['val']:
                left = True

        if left:
            if isinstance(node['left'], dict):
                return self.__predict_sample(node['left'], sample)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.__predict_sample(node['right'], sample)
            else:
                return node['right']

    # Gini index (Perfect is 0)
    def __gini_index(self, groups, classes):
        gini = 0.0
        n_samples = sum([g.shape[0] for g in groups])
        for g in groups:
            if g.shape[0] > 0:
                score = 1.0
                # score the group based on each class
                for c in classes:
                    p = np.count_nonzero(g[:, -1] == c) / g.shape[0]
                    score -= p ** 2
                # weight the group score by size
                gini += (g.shape[0] / n_samples) * score
        return gini

    def __test_split(self, X, idx, value):
        if idx in self.numerical_idx:
            # left_idx = np.where(X[:, idx] < value)[0]
            left_idx = X[:, idx] < value
        else:
            left_idx = X[:, idx] == value
        right_idx = np.logical_not(left_idx)
        return X[left_idx, :], X[right_idx, :]

    def __split(self, node, depth):
        left = np.copy(node['groups'][0])
        right = np.copy(node['groups'][1])
        del (node['groups'])
        if len(left) < 1 or len(right) < 1:
            node['left'] = node['right'] = self.__terminal(np.vstack((left, right)))
            return
        if 0 < self.max_depth <= depth:
            node['left'], node['right'] = self.__terminal(left), self.__terminal(right)
            return

        if len(left) <= self.min_size:
            node['left'] = self.__terminal(left)
        else:
            node['left'] = self.__get_split(left)
            self.__split(node['left'], depth + 1)

        if len(right) <= self.min_size:
            node['right'] = self.__terminal(right)
        else:
            node['right'] = self.__get_split(right)
            self.__split(node['right'], depth + 1)

    def __get_split(self, X):
        classes = np.unique(X[:, -1])
        # Include labels to make it easier

        b_idx, b_val, b_gini, b_groups = 9999, 9999, 9999, None
        f_idx = np.arange(X.shape[1] - 1)
        if isinstance(self.subsample_size, str):
            f_idx = np.random.choice(f_idx, size=int(np.random.uniform(1, X.shape[1]-1)), replace=False)
        elif self.subsample_size > 0:
            f_idx = np.random.choice(f_idx, size=self.subsample_size, replace=False)

        for idx in f_idx:
            # When categorical no need to go row by row
            for v in np.unique(X[:, idx]):
                groups = self.__test_split(X, idx, v)
                gini = self.__gini_index(groups, classes)
                if gini < b_gini:
                    b_idx, b_val, b_gini, b_groups = idx, v, gini, groups
        if b_groups[0].shape[0] > 1 and b_groups[1].shape[0] > 0:
            self.rule_counts[b_idx] = self.rule_counts.get(b_idx, {})
            self.rule_counts[b_idx][b_val] = self.rule_counts[b_idx].get(b_val, 0) + 1

        return {'idx': b_idx, 'val': b_val, 'groups': b_groups}

    def __terminal(self, group):
        (v, c) = np.unique(group[:, -1], return_counts=True)
        return v[np.argmax(c)]


