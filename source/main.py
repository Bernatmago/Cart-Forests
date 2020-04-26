from sklearn.metrics import classification_report

from cart import CART
from forest import RandomForest, DecisionForest
from data_loader import load_data


X_train, X_test, y_train, y_test, numerical_idx = load_data('heart.csv')


n_trees = 100
s_bootstrap = int(X_train.shape[0]/1.25)
n_features = int(X_train.shape[1]/1.25)

# Check End criteria

# f = RandomForest(n_trees, s_bootstrap, n_features, max_depth=5, min_size=20)
# f.fit(X_train, y_train, numerical_idx)
# y_pred = f.predict(X_test)
# print(classification_report(y_test, y_pred))

f = DecisionForest(n_trees, n_features, max_depth=5, min_size=1)
f.fit(X_train, y_train, numerical_idx)
y_pred = f.predict(X_test)
print(classification_report(y_test, y_pred))

# c = CART(max_depth=5, min_size=20, subsample_size=n_features)
# c.fit(X_train, y_train, numerical_idx)
# y_pred = c.predict(X_test)
# print(classification_report(y_test, y_pred))
# print(c)


