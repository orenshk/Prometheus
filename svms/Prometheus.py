import time
import numpy as np
import os
import sys
from sklearn import svm
from scipy.spatial.distance import cdist

def merge(arrays, idx_to_omit):
    """
    Merge the np.arrays in a list into a single 2D np.array, omitting the row at idx.
    (len(arrays) - 1) * arrays[0].shape[0] by arrays[0].shape[1]

    :type arrays: list
    :param arrays: list of np.arrays to merge
    :param idx_to_omit: index of row to omit.
    :return: merged np.array.
    """

    result = np.array(arrays[0:idx_to_omit] + arrays[idx_to_omit+1:])
    result.shape = (result.shape[0] * result.shape[1], result.shape[2])
    return result


# location of data
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

print("Loading data", end='')
# load the data into a numpy array.
with open(os.path.join(DATA_DIR, 'training.csv'), 'r') as training_file:
    data_rows = training_file.readlines()

# number of examples (250000)
N = len(data_rows) - 1  # first row of training.csv is column names
# number of features (30)
D = len(data_rows[0].strip().split(',')) - 3  # ignore the id, weight, and label values.

# data is an N x D array
all_train_data = np.zeros((N, D))

# weight vector
all_train_weights = np.zeros((N, 1))

# label vector.
all_train_labels = np.zeros((N, 1), dtype=str)

# process the rows, and fill up the data and labels. Ignore the first row.
for i, row in enumerate(data_rows[1:]):
    values = row.strip().split(',')

    # The features are indexed one (inclusive) to the second to last (exclusive).
    all_train_data[i] = values[1:-2]

    # the weight is in the second to last column
    all_train_weights[i] = values[-2]

    # the label is in the last column
    all_train_labels[i] = values[-1]

    # show some progress
    if i > 0 and i % 10000 == 0:
        print('.', end='')
        sys.stdout.flush()

print('done')

# Train svm.
kernel = 'sigmoid'
n_folds = 25

# size of the data used for training.
M = N - (N / n_folds)


if kernel == 'precomputed':
    print('kernalizing...', end='')
    s = 0.1
    all_data_gram = np.exp(-cdist(all_train_data, all_train_data) / (2 * s ** 2))
    print('done')
    split_train_data = np.array_split(all_data_gram, n_folds)
else:
    split_train_data = np.array_split(all_train_data, n_folds)

# split the data in n_folds chunks.
split_train_weights = np.array_split(all_train_weights, n_folds)
split_train_labels = np.array_split(all_train_labels, n_folds)

# each chunk will be used as a test vector
for i, test_data in enumerate(split_train_data):
    test_weights = split_train_weights[i]
    test_labels = split_train_labels[i]

    # The other chunks will be the training data.
    train_data = merge(split_train_data, idx_to_omit=i)
    train_weights = merge(split_train_weights, idx_to_omit=i)
    train_labels = merge(split_train_labels, idx_to_omit=i)

    classifier = svm.SVC(kernel=kernel)
    print(classifier)
    st_time = time.time()
    print('fold number {} out of {}'.format(i, n_folds))
    print('fitting...', end='')
    classifier.fit(train_data, train_labels.flat)
    print('elapsed training time: '.format(time.time() - st_time))

    # compare to test_data
    predicted_labels = classifier.predict(test_data)
    num_correct = np.sum(predicted_labels == test_labels.flat)
    print('num errors: {}'.format(num_correct))
