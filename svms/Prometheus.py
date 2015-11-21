import numpy as np
import os
import sys
import argparse
import pickle
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from utils.notification import email, push_note


def merge(arrays, idx_to_omit):
    """
    Merge the np.arrays in a list into a single 2D np.array, omitting the row at idx.
    (len(arrays) - 1) * arrays[0].shape[0] by arrays[0].shape[1]

    :type arrays: list
    :param arrays: list of np.arrays to merge
    :param idx_to_omit: index of row to omit.
    :return: merged np.array.
    """

    result = np.concatenate(arrays[0:idx_to_omit] + arrays[idx_to_omit+1:])
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Prometheus svm')
    parser.add_argument('--email', nargs='*', default='')
    parser.add_argument('--test_size', type=float, default=0.5)
    parser.add_argument('--data_file', default='training.csv')
    parser.add_argument('--push', action='store_true')

    args = parser.parse_args()

    # location of data
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    print("Loading data", end='')
    # load the data into a numpy array.
    with open(os.path.join(DATA_DIR, args.data_file), 'r') as training_file:
        data_rows = training_file.readlines()

    # number of examples (250000)
    N = len(data_rows) - 1  # first row of training.csv is column names
    # number of features (30 + 1 weight)
    D = len(data_rows[0].strip().split(',')) - 2  # ignore the id and the label. Keep the weight for now.

    # data is an N x D array
    all_data = np.zeros((N, D))

    # label vector.
    all_labels = np.zeros((N, 1), dtype=str)

    # process the rows, and fill up the data and labels. Ignore the first row.
    for i, row in enumerate(data_rows[1:]):
        values = row.strip().split(',')

        all_data[i] = values[1:-1]

        all_labels[i] = values[-1]

        # show some progress
        if i > 0 and i % 10000 == 0:
            print('.', end='')
            sys.stdout.flush()

    print('done')

    # we're going to use grid search to find best parameters.
    train_data, test_data, train_labels, test_labels = train_test_split(all_data, all_labels, test_size=args.test_size)

    parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1e3]},
                  {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1e3], 'coef0': [0, 1]}]

    # separate out the weights.
    train_weights = train_data[:, -1]
    train_data = train_data[:, 0:-1]

    # no need to save the test weights. Right?
    test_data = test_data[:, 0:-1]

    classifier = GridSearchCV(svm.SVC(cache_size=2500), parameters, cv=3, fit_params={'sample_weight': train_weights.flat})
    print('starting search. This may be a while...')
    classifier.fit(train_data, train_labels.flat)

    # best parameters
    print('best parameters: {}'.format(classifier.best_params_))

    for params, mean_score, scores in classifier.grid_scores_:
        print('{:.3f} (+/-{:.03f}) for {}'.format(mean_score, scores.std() * 2, params))

    print('classification report')
    report = classification_report(y_true=test_labels, y_pred=classifier.predict(test_data))
    print(report)

    if args.email:
        body = report
        email(to_addrs=args.email, subject='prometheus done training', body=report)

    if args.push:
        push_note(subject='prometheus done training', body=report)

    # save the classifier.
    with open('classifier-{}.dat'.format(classifier.best_params_), 'wb') as f:
        pickle.dump(classifier, f, protocol=3)
