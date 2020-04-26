import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filename, test_size=0.33):
    df = pd.read_csv('../data/' + filename)
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    numerical_idx = []
    # 10 esta en dubte crec
    if filename == 'heart.csv': numerical_idx = [0, 3, 4, 7, 9, 10, 11]
    if filename == 'semeion.csv': numerical_idx = []
    if filename == 'satimage.csv': numerical_idx = range(X.shape[1])

    return X_train, X_test, y_train, y_test, numerical_idx
