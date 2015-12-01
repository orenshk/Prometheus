import pickle
import os
import sys
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def load_classifier(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def load_data(data_file):

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
    labels = np.zeros((N, 1), dtype=str)

    # process the rows, and fill up the data and labels. Ignore the first row.
    for i, row in enumerate(data_rows[1:]):
        values = row.strip().split(',')

        data[i] = values[1:-1]

        labels[i] = values[-1]

        # show some progress
        if i > 0 and i % 10000 == 0:
            print('.', end='')
            sys.stdout.flush()

    print('done')
    sys.stdout.flush()
    return data, labels
