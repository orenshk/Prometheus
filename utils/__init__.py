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


def load_data(data_file, test_size, one_of_k=False):
    """
    Load the data from data_file and split it into training set and test set. The test size is a fraction of the set.
    :param one_of_k: return labels as 1-of-k vectors
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
    N = len(data_rows) - 1  # first row of training.csv is column names
    # number of features (30 + 1 weight)
    D = len(data_rows[0].strip().split(',')) - 2  # ignore the id and the label. Keep the weight for now.

    # data is an N x D array
    data = np.zeros((N, D))

    # label vector.
    if one_of_k:
        labels = np.zeros((N, 2), dtype=float)
    else:
        labels = np.zeros((N, 1), dtype=str)

    # process the rows, and fill up the data and labels. Ignore the first row.
    for i, row in enumerate(data_rows[1:]):
        values = row.strip().split(',')

        data[i] = values[1:-1]

        if one_of_k:
            if values[-1] == 's':
                labels[i] = np.array([1, 0])
            else:
                labels[i] = np.array([0, 1])
        else:
            labels[i] = values[-1]

        # show some progress
        if i > 0 and i % 10000 == 0:
            print('.', end='')
            sys.stdout.flush()

    print('done')
    sys.stdout.flush()

    # we're going to use grid search to find best parameters.
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_size)

    # separate out the weights.
    train_weights = train_data[:, -1]
    train_data = train_data[:, 0:-1]

    # no need to save the test weights. Right?
    test_data = test_data[:, 0:-1]

    return train_data, test_data, train_weights, train_labels, test_labels
