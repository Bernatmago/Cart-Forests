from sklearn.metrics import classification_report, accuracy_score, f1_score

from forest import RandomForest, DecisionForest
from data_loader import load_data, plot_grid_search
import operator
import pprint
datasets = ['heart', 'semeion', 'mushrooms']
for dataset in datasets:
    X_train, X_test, y_train, y_test, numerical_idx, names = load_data(dataset + '.csv')
    n_trees = [1, 10, 25, 50, 75, 100]
    s_bootstrap = int(X_train.shape[0] / 2)
    n_features = [int(X_train.shape[1] / 4), int(X_train.shape[1] / 2), int(3 * X_train.shape[1] / 4), 'r_uniform']
    fo = open('out_report_{}.txt'.format(dataset), 'w')
    fr = open('out_rules_{}.txt'.format(dataset), 'w')
    accs = []
    f1s = []
    u = []
    for algorithm in ['df', 'rf']:
        for f in n_features:
            for n in n_trees:
                if algorithm == 'df':
                    forest = DecisionForest(n, f, max_depth=-1, min_size=-1)
                else:
                    forest = RandomForest(n, s_bootstrap, f, max_depth=-1, min_size=1)
                forest.fit(X_train, y_train, numerical_idx)
                y_pred = forest.predict(X_test)
                accs.append(accuracy_score(y_test, y_pred))
                f1s.append(f1_score(y_test, y_pred, average='macro'))
                fo.write('{}_{}_{}_{}\n'.format(dataset, n, f, algorithm))
                fo.write(classification_report(y_test, y_pred))
                fo.write('\n')
                fr.write('{}_{}_{}_{}\n'.format(dataset, n, f, algorithm))
                rc = forest.rule_count()
                keys = list(rc.keys())
                for k in keys:
                    c = 0
                    for kk in rc[k].keys():
                        c += rc[k][kk]
                    rc[k] = c
                    rc[names[k]] = rc[k]
                    del rc[k]
                rc = ['{}: {}'.format(k, v) for k, v in sorted(rc.items(), key=operator.itemgetter(1), reverse=True)]
                pprint.pprint(rc, fr)
                fr.write('\n')
                print('{}_{}_{}_{}'.format(dataset, n, f, algorithm))
            u.append('{}_{}'.format(algorithm, f))
    plot_grid_search(accs, n_trees, u, 'Trees', 'Conf', 'Accuracy', dataset)
    plot_grid_search(f1s, n_trees, u, 'Trees', 'Features', 'F1_macro', dataset)
    fo.close()
    fr.close()

