# Decision & Random Forests using CART
Simple python implementation for generating random or decision forests using the CART (Classification and Regression Trees) algorithm to generate each tree.

Information regarding the forests:
* Decision Forests: https://ieeexplore.ieee.org/document/709601
* Random Forests: https://link.springer.com/article/10.1023/A:1010933404324

Datasets used for experimentation:
* Hearth Disease: https://www.kaggle.com/ronitf/heart-disease-uci
* Semeion Handwritten Digits http://archive.ics.uci.edu/ml/datasets/Semeion+Handwritten+Digit
* Satimage (satlog): http://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29

To run the defined experiments just execute main.py


### Decision Forest Example:
```
from forest import DecisionForest
forest = RandomForest(n_trees, n_features, max_depth=-1, min_size=1)
forest.fit(X_train, y_train, numerical_attr_idx)
y_pred = forest.predict(X_test)

```


### Random Forest Example:
```
from forest import RandomForest
forest = RandomForest(n_trees, size_bootstrap, n_features, max_depth=-1, min_size=1)
forest.fit(X_train, y_train, numerical_attr_idx)
y_pred = forest.predict(X_test)

```

### CART Tree Example:
```
from cart import CART
tree = CART(max_depth=-1, min_size=1, subsample_size=-1)
tree.fit(X_train, y_train, numerical_attr_idx)
y_pred = tree.predict(X_test)

```
