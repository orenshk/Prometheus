from __future__ import print_function
import pickle
import os
import sys
import numpy as np
from sklearn.cross_validation import train_test_split

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_classifier(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_data(data_file, test_size, encoding='string', normalize=False):
    """
    Load the data from data_file and split it into training set and test set. The test size is a fraction of the set.
    :type encoding: str
    :param encoding: type of encoding to use for classes.
        string will keep 's' and 'b' labels.
        one-hot will make two vectors, (1,0) for s, (0, 1) for b
        integer will assign integer classes - s: 0, b: 1
    :param data_file: name of data file. Assumed to be in DATA_DIR
    :param test_size: float, proprtion of data to be used for test.
    :return: tuple containing:
        train_data: ndarray of shape N * (1 - test_size) by D
        test_data: ndarray of shape test_size * N by D
        test_weights: ndarray of shape N * (1 - test_size) by 1
        train_labels: ndarray of shape N * (1 - test_size) by 1
        test_labels: ndarray of shape test_size * N by 1
    """

    print("Loading data", end='')
    # load the data into a numpy array.
    with open(os.path.join(DATA_DIR, data_file), 'r') as training_file:
        data_rows = training_file.readlines()

    # number of examples (250000)
    n = len(data_rows) - 1  # first row of training.csv is column names
    # number of features (30 + 1 weight)
    d = len(data_rows[0].strip().split(',')) - 2  # ignore the id and the label. Keep the weight for now.

    # data is an N x D array
    data = np.zeros((n, d))

    labels, label_map = _init_label_vector(encoding, n)

    # process the rows, and fill up the data and labels. Ignore the first row.
    for i, row in enumerate(data_rows[1:]):
        values = row.strip().split(',')

        data[i] = values[1:-1]
        labels[i] = label_map[values[-1]]

        # show some progress
        if i > 0 and i % 10000 == 0:
            print('.', end='')
            sys.stdout.flush()

    print('done')
    sys.stdout.flush()

    if normalize:
        mean_data = np.mean(data, 0)
        std_data = np.std(data, 0)
        data = (data - mean_data) / std_data

    # we're going to use grid search to find best parameters.
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size)

    # separate out the weights.
    train_weights = train_data[:, -1]
    train_data = train_data[:, 0:-1]

    # no need to save the test weights. Right?
    test_data = test_data[:, 0:-1]

    return train_data, test_data, train_weights, train_labels, test_labels


def _init_label_vector(encoding, n):
    """
    initialize the label vector.

    :param encoding:  type of encoding to use. Either string, integer, or one-hot
    :param n: number of samples in set.
    :return: an np.array containing all zeros, of the required type, and a dictionary mapping labels to encoding.
    """
    # label vector.
    if encoding == 'one-hot':
        labels = np.zeros((n, 2), dtype=float)
        label_map = {'s': [1, 0], 'b': [0, 1]}
    elif encoding == 'integer':
        labels = np.zeros((n,), dtype=int)
        label_map = {'s': 0, 'b': 1}
    elif encoding == 'string':
        labels = np.zeros((n, 1), dtype=str)
        label_map = {'s': 's', 'b': 'b'}
    else:
        raise TypeError('unsupported encoding: {}'.format(encoding))

    return labels, label_map
