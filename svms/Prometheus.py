import numpy as np
import os
from sklearn import svm

# location of data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


def train(n_folds=25):
    """
    Train svm using cross validation on n_folds.
    :type n_folds: int
    :param n_folds: number of folds to use in cross validation. Set to 1 to use entire data set.
    """


# load the data into a numpy array.
with open(os.path.join(DATA_DIR, 'training.csv'), 'r') as training_file:
    data_rows = training_file.readlines()

# number of examples (250000)
N = len(data_rows) - 1  # first row of training.csv is column names
# number of features (30)
D = len(data_rows[0].strip().split(',')) - 3  # ignore the id, weight, and label values.

# data is an N x D array
train_data = np.zeros((N, D))

# weight vector
train_weights = np.zeros((N, 1))

# label vector. We'll keep this as a regular list.
train_labels = []

# process the rows, and fill up the data and labels. Ignore the first row.
for i, row in enumerate(data_rows[1:]):
    values = row.strip().split(',')

    # The features are indexed one (inclusive) to the second to last (exclusive).
    train_data[i] = values[1:-2]

    # the weight is in the second to last column
    train_weights[i] = values[-2]

    # the label is in the last column
    train_labels.append(values[-1])

