import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
plt.style.use('ggplot')


def load_data(filename, test_size=0.33):
    df = pd.read_csv('../data/' + filename)
    X = df.to_numpy()[:, :-1]
    y = df.to_numpy()[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    numerical_idx = []
    # 10 esta en dubte crec
    if filename == 'heart.csv': numerical_idx = [0, 3, 4, 7, 9, 11]
    if filename == 'semeion.csv': numerical_idx = []
    if filename == 'satimage.csv': numerical_idx = range(X.shape[1])
    if filename == 'mushrooms.csv': numerical_idx = []

    return X_train, X_test, y_train, y_test, numerical_idx, list(df.columns.values)


def plot_grid_search(results, p1, p2, name1, name2, metric, dataset):
    scores = np.array(results).reshape(len(p2), len(p1))

    _, ax = plt.subplots(1, 1)
    for i, v in enumerate(p2):
        ax.plot(p1, scores[i, :], '-o', label=name2 + ': ' + str(v))
    ax.set_title("{} Grid Search Scores".format(dataset), fontsize=15, fontweight='bold')
    ax.set_xlabel(name1, fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.legend(loc='best', fontsize=8)
    ax.grid('on')
    plt.savefig('../plots/{}_{}_{}_{}.png'.format(dataset, metric, name1, name2))


