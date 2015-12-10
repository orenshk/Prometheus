import os
import gzip
import pickle
import timeit
import argparse
import numpy as np
import sys
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


def load_higgs_data(data_file, valid_size, normalize):
    # we get back a tuple of train data, test data, train weights, train labels, and test labels
    dataset = load_data(data_file, valid_size, encoding='integer', normalize=normalize)

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
                     cost,
                     train_set_x,
                     train_set_y,
                     valid_set_x,
                     valid_set_y,
                     x,
                     y,
                     learning_rate=0.1,
                     n_epochs=1000,
                     batch_size=1):
    """
    Run stochastic gradient descent for the given clssifier and training data.

    :param Classifier classifier:  classifier to be trainedx
    :param function cost: cost function used in training.
    :param theano.shared train_set_x: training data set.
    :param theano.shared train_set_y: training labels.
    :param theano.shared valid_set_x: validation set.
    :param theano.shared valid_set_y: validation labels.
    :param theano.matrix x: variable to hold feature data.
    :param theano.matrix y: variable to hold label data.
    :param float learning_rate:
    :param int n_epochs:
    :param int batch_size:
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

    return do_training(training_step=train_model,
                       training_step_args=[],
                       validation_step=validate_model,
                       n_epochs=n_epochs,
                       n_train_batches=n_train_batches,
                       n_valid_batches=n_valid_batches)


def adv_sgd(classifier,
            cost,
            train_set_x,
            train_set_y,
            valid_set_x,
            valid_set_y,
            x,
            y,
            learning_rate=0.05,
            n_epochs=1000,
            batch_size=1):
    """
    Run stochastic gradient descent for the given classifier and training data, using
    adversarial regularization

    :param Classifier classifier:  classifier to be trained
    :param function cost: cost function used in training.
    :param theano.shared train_set_x: training data set.
    :param theano.shared train_set_y: training labels.
    :param theano.shared valid_set_x: validation set.
    :param theano.shared valid_set_y: validation labels.
    :param theano.matrix x: variable to hold feature data.
    :param theano.matrix y: variable to hold label data.
    :param float learning_rate:
    :param int n_epochs:
    :param int batch_size:
    """

    # compute number of minibatches for training, validation and testing
    n_train_batches = int(train_set_x.get_value(borrow=True).shape[0] / batch_size)
    n_valid_batches = int(valid_set_x.get_value(borrow=True).shape[0] / batch_size)

    index = T.lscalar()

    # function that will validate the model during training
    validate_model = th.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # the gradients for the network:
    # LogisticRegression: W, b
    # Multilayer: hidden.W, hidden.b, output.W, output.b
    grads = [T.grad(cost=cost, wrt=param) for param in classifier.params]

    # the gradient of the cost function w.r.t x, for the adversarial part.
    grads.append(T.grad(cost=cost, wrt=x))

    # we need the signs of the gradient w.r.t x
    grad_x = grads[-1]
    sign_grad_x = (grad_x > 0) - (grad_x < 0)

    updates = [
        (param, param - learning_rate * grad)
        for param, grad in zip(classifier.params, grads)
        ]

    # in the first run, we'll get the gradients.
    first_run = th.function(
        inputs=[index],
        outputs=sign_grad_x,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # here we'll get the gradient w.r.t. W & b of the cost function applied to x + sign(grad).
    eps = 0.001
    sign_grad = T.matrix('sign_grad')
    second_run = th.function(
        inputs=[index, sign_grad],
        outputs=grads,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size] + eps * sign_grad,
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # now it's time for the train step. We'll define the updates for every grad, except the grad w.r.t x
    return do_training(training_step=adv_training_steps,
                       training_step_args=[first_run, second_run],
                       validation_step=validate_model,
                       n_epochs=n_epochs,
                       n_train_batches=n_train_batches,
                       n_valid_batches=n_valid_batches)


def adv_training_steps(index, first_run, second_run):
    sign_grad_x = first_run(index)
    second_run(index, sign_grad_x)


def do_training(training_step, training_step_args, validation_step, n_epochs, n_train_batches, n_valid_batches):
    print('n_epochs: {}, n_train_batches: {}'.format(n_epochs, n_train_batches))

    validation_frequency = 10000

    best_validation_loss = np.inf
    best_validation_losses = []
    best_validation_life = 11
    improvement_threshold = best_validation_life * 1e3

    start_time = timeit.default_timer()

    epoch = 0
    global_iter_num = 0

    while epoch < n_epochs:
        epoch += 1
        for minibatch_index in range(n_train_batches):
            training_step(minibatch_index, *training_step_args)
            global_iter_num += 1
            if minibatch_index % validation_frequency == 0:
                sys.stdout.flush()

                best_validation_loss = validate(validation_step,
                                                best_validation_loss,
                                                epoch,
                                                minibatch_index,
                                                n_valid_batches,
                                                global_iter_num)

        print('done epoch {}. Elapsed time: {:.3f}'.format(epoch, timeit.default_timer() - start_time))
        print('best validation score for epoch was: {}'.format(best_validation_loss * 100))

        # check if we should stop earlier. If there hasn't been a change in the last few generations, stop.
        best_validation_losses.append(best_validation_loss)
        if len(best_validation_losses) == best_validation_life:
            if np.all(np.diff(best_validation_losses) <= improvement_threshold):
                print('no improvement for the last {} epochs. Stopping.'.format(best_validation_life))
                break
            else:
                best_validation_losses = best_validation_losses[1:]

    end_time = timeit.default_timer()
    print('Optimization complete with best validation score of {}'.format(best_validation_loss * 100.))
    elapsed = end_time - start_time
    print('The code ran for {} epochs, with {} epochs / sec'.format(epoch, 1. * epoch / elapsed))
    print('Total run time was: {:.3f} seconds'.format(elapsed))

    return best_validation_loss


def validate(validation_step,
             best_validation_loss,
             epoch,
             minibatch_index,
             n_valid_batches,
             global_iter_num):

    improvement_threshold = 0.995

    validation_losses = [validation_step(i) for i in range(n_valid_batches)]
    this_validation_loss = np.mean(validation_losses)

    print(
        'epoch {}, minibatch {}, validation error {}'.format(epoch,
                                                             minibatch_index,
                                                             this_validation_loss * 100.)
    )

    # if there has been an improvement in the best validation loss score, we record it.
    if this_validation_loss < best_validation_loss:
        validation_loss_improvement = best_validation_loss * improvement_threshold - this_validation_loss
        if 0 < validation_loss_improvement < np.inf:
            print('Improved validation loss: {}'.format(validation_loss_improvement))

        best_validation_loss = this_validation_loss

    return best_validation_loss


def main(argv, datasets=None, num_classes=None):
    if datasets:
        assert num_classes is not None, "num_classes must be provided if the dataset isn't loaded from file"
    else:
        if argv.problem == 'mnist':
            datasets = load_mnist_data()
            num_classes = 10
        else:
            datasets = load_higgs_data(data_file=argv.data_file, valid_size=argv.valid_size, normalize=argv.normalize)
            num_classes = 2

    # get dimensionality and number of classes.
    dim = datasets[0][0].get_value(borrow=True).shape[1]

    x = T.matrix('x')
    y = T.ivector('y')

    n_hidden = argv.n_hidden
    L1_reg = 0.0
    L2_reg = 0.01
    clf = MultilayerPerceptron(input_data=x, n_in=dim, n_hidden=n_hidden, n_out=num_classes, cost=argv.cost)
    cost = clf.cost(y) + L1_reg * clf.L1 + L2_reg * clf.L2

    if argv.adv:
        result = adv_sgd(classifier=clf,
                         cost=cost,
                         train_set_x=datasets[0][0],
                         train_set_y=datasets[0][1],
                         valid_set_x=datasets[1][0],
                         valid_set_y=datasets[1][1],
                         x=x,
                         y=y,
                         n_epochs=argv.n_epochs)
    else:
        result = sgd_optimization(classifier=clf,
                                  cost=cost,
                                  train_set_x=datasets[0][0],
                                  train_set_y=datasets[0][1],
                                  valid_set_x=datasets[1][0],
                                  valid_set_y=datasets[1][1],
                                  x=x,
                                  y=y,
                                  n_epochs=argv.n_epochs)

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', default='mnist', choices={'mnist', 'higgs'})
    parser.add_argument('--data_file', default='training.csv')
    parser.add_argument('--valid_size', default=0.1, type=float)
    parser.add_argument('--n_epochs', default=2000, type=int)
    parser.add_argument('--n_hidden', default=600, type=int)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--normalize', action='store_true')
    parser.add_argument('--cost', default='neg_log', choices={'neg_log', 'cross_ent'})

    args = parser.parse_args()
    main(args)
