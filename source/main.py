from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from cart import CART
from forest import RandomForest, DecisionForest

# df = pd.read_csv('../data/banknotes.txt')

df = pd.read_csv('../data/mushrooms.csv')
X = df.to_numpy()[:, :-1]
y = df.to_numpy()[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
n_trees = 10
s_bootstrap = 100
n_features = 10
f = RandomForest(n_trees, s_bootstrap, n_features, max_depth=10, min_size=6)
f.fit(X_train, y_train)
f = DecisionForest(n_trees, n_features)
f.fit(X_train, y_train)



# c = CART(max_depth=10, min_size=5, subsample_size=10)
# a = c.fit(X_train, y_train)
# y_pred = c.predict(X_test)
# print(classification_report(y_test, y_pred))
# print('a')
# Check End criteria

# All rows are in the same class
# Si se cumple categoria go left else go right
