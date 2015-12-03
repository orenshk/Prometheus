import os
import gzip
import pickle
import timeit
import argparse
import numpy as np
import theano as th
import theano.tensor as T
from utils import load_data
from theano_mlp import MultilayerPerceptron
from theano_logistic_regression import LogisticRegression


def load_mnist_data():
    data_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data',
                            'MNIST_data',
                            'mnist.pkl3.gz')

    with gzip.open(data_loc, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)

    test_set_x, test_set_y = load_shared_dataset(test_set)
    valid_set_x, valid_set_y = load_shared_dataset(valid_set)
    train_set_x, train_set_y = load_shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_higgs_data(data_file, valid_size):
    # we get back a tuple of train data, test data, train weights, train labels, and test labels
    dataset = load_data(data_file, valid_size, encoding='integer')

    train_set_x, train_set_y = load_shared_dataset((dataset[0], dataset[3]))
    valid_set_x, valid_set_y = load_shared_dataset((dataset[1], dataset[4]))

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]


def load_shared_dataset(data_xy, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.

    :type borrow: bool
    :param borrow:
    :type data_xy: 2-uple of numpy.arrays. One of shape (N, D), one of shape (N,)
    :param data_xy: dataset to be loaded into shared variables.
    """
    data_x, data_y = data_xy
    shared_x = th.shared(np.asarray(data_x,
                                    dtype=th.config.floatX),
                         borrow=borrow)
    shared_y = th.shared(np.asarray(data_y,
                                    dtype=th.config.floatX),
                         borrow=borrow)

    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets us get around this issue
    return shared_x, T.cast(shared_y, 'int32')


def sgd_optimization(classifier,
                     train_set_x,
                     train_set_y,
                     valid_set_x,
                     valid_set_y,
                     learning_rate=0.13,
                     n_epochs=1000,
                     batch_size=600):
    """
    Run stochastic gradient descent for the given clssifier and training data.

    :param Classifier classifier:  classifier to be trained
    :param theano.shared train_set_x: training data set.
    :param theano.shared train_set_y: training labels
    :param theano.shared valid_set_x: validation set
    :param theano.shared valid_set_y: validation labels
    :param float learning_rate:
    :param int n_epochs:
    :param int batch_size:
    :return:
    """

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0] / batch_size)

    index = T.lscalar()

    validate_model = th.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    grads = [T.grad(cost=cost, wrt=param) for param in classifier.params]
    updates = [
        (param, param - learning_rate * grad)
        for param, grad in zip(classifier.params, grads)
        ]

    train_model = th.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    patience = 5000
    patience_increase = 2

    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches, patience / 2)

    best_validation_loss = np.inf
    start_time = timeit.default_timer()

    done_looping = False
    epoch = 0
    global_iter_num = 0
    while epoch < n_epochs and not done_looping:
        epoch += 1
        for minibatch_index in range(n_train_batches):
            train_model(minibatch_index)

            if (global_iter_num + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print(
                    'epoch {}, minibatch {}/{}, validation error {}'.format(epoch,
                                                                            minibatch_index + 1,
                                                                            n_train_batches,
                                                                            this_validation_loss * 100.)
                )

                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, global_iter_num * patience_increase)

                    best_validation_loss = this_validation_loss

                    # save the best model
                    with open('best_model.pkl', 'wb') as f:
                        pickle.dump(classifier, f)

            if patience <= global_iter_num:
                done_looping = True

            global_iter_num += 1

    end_time = timeit.default_timer()
    print(
        'Optimization complete with best validation score of {}'.format(best_validation_loss * 100.)
    )
    elapsed = end_time - start_time
    print('The code ran for {} epochs, with {} epochs / sec'.format(epoch, 1. * epoch / elapsed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='mnist', choices={'mnist', 'higgs'})
    parser.add_argument('--data_file', default='training.csv')
    parser.add_argument('--valid_size', default=0.1, type=float)
    parser.add_argument('--n_epochs', default=2000, type=int)
    parser.add_argument('--classifier', default='LR', choices={'LR', 'MLP'})
    parser.add_argument('--n_hidden', default=500, type=int)

    args = parser.parse_args()

    if args.problem == 'mnist':
        datasets = load_mnist_data()
        num_classes = 10
    else:
        datasets = load_higgs_data(data_file=args.data_file, valid_size=args.valid_size)
        num_classes = 2

    # get dimensionality and number of classes.
    dim = datasets[0][0].get_value(borrow=True).shape[1]

    x = T.matrix('x')
    y = T.ivector('y')

    # initialize the classifier.
    if args.classifier == 'LR':
        clf = LogisticRegression(input_data=x, n_in=dim, n_out=num_classes)
        cost = clf.cost(y)
    elif args.classifier == 'MLP':
        n_hidden = args.n_hidden
        L1_reg = 0.0
        L2_reg = 0.01
        clf = MultilayerPerceptron(input_data=x, n_in=dim, n_hidden=n_hidden, n_out=num_classes)
        cost = clf.cost(y) + L1_reg * clf.L1 + L2_reg * clf.L2
    else:
        raise NotImplementedError('Unsupported classifier')

    sgd_optimization(classifier=clf,
                     train_set_x=datasets[0][0],
                     train_set_y=datasets[0][1],
                     valid_set_x=datasets[1][0],
                     valid_set_y=datasets[1][1])
