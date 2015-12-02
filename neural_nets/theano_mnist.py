import os
import gzip
import theano as th
import theano.tensor as T
import numpy as np
import cPickle
import timeit
from utils import load_data


class LogisticRegression(object):
    def __init__(self, input_data, n_in, n_out):
        # initialize the weights
        self.W = th.shared(
            value=np.zeros((n_in, n_out), dtype=th.config.floatX),
            name='W',
            borrow=True
        )

        # initialize the bias
        self.b = th.shared(
            value=np.zeros((n_out, ), dtype=th.config.floatX),
            name='b',
            borrow='True'
        )

        self.input = input_data

        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        assert y.dtype.startswith('int')

        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 'y', y.type, 'y_pred', self.y_pred.type)

        return T.mean(T.neq(self.y_pred, y))


def load_mnist_data():
    data_loc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'data',
                            'MNIST_data',
                            'mnist.pkl.gz')

    with gzip.open(data_loc, 'rb') as f:
        train_set, valid_set, test_set = cPickle.load(f)

    test_set_x, test_set_y = load_shared_dataset(test_set)
    valid_set_x, valid_set_y = load_shared_dataset(valid_set)
    train_set_x, train_set_y = load_shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


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


def sgd_optimization_mnist(learning_rate=0.13, n_epochs=1000, batch_size=600):

    datasets = load_mnist_data()
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    do_training(learning_rate, n_epochs, batch_size, train_set_x, train_set_y, valid_set_x, valid_set_y)


def do_training(learning_rate, n_epochs, batch_size,
                train_set_x, train_set_y, valid_set_x, valid_set_y, num_classes=10):

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size

    # get dimensionality and number of classes.
    dim = train_set_x.get_value(borrow=True).shape[1]

    index = T.lscalar()

    x = T.matrix('x')
    y = T.ivector('y')

    classifier = LogisticRegression(input_data=x, n_in=dim, n_out=num_classes)

    cost = classifier.negative_log_likelihood(y)

    validate_model = th.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    g_W = T.grad(cost=cost, wrt=classifier.W)
    g_b = T.grad(cost=cost, wrt=classifier.b)

    updates = [(classifier.W, classifier.W - learning_rate * g_W),
               (classifier.b, classifier.b - learning_rate * g_b)]

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
        for minibatch_index in xrange(n_train_batches):
            train_model(minibatch_index)

            if (global_iter_num + 1) % validation_frequency == 0:
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
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
                        cPickle.dump(classifier, f)

            if patience <= global_iter_num:
                done_looping = True

            global_iter_num += 1

    end_time = timeit.default_timer()
    print(
        'Optimization complete with best valdiation score of {}'.format(best_validation_loss * 100.)
    )
    elapsed = end_time - start_time
    print('The code ran for {} epochs, with {} epochs / sec'.format(epoch, 1. * epoch / elapsed))

if __name__ == '__main__':
    sgd_optimization_mnist()