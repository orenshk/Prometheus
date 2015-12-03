import theano as th
import theano.tensor as T
import numpy as np


class LogisticRegression(object):
    def __init__(self, input_data, n_in, n_out, cost=None):

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

        # input data for the regression. This is x.
        self.input = input_data

        # as always, we do softmax over the Wx + b.
        self.p_y_given_x = T.nnet.softmax(T.dot(self.input, self.W) + self.b)

        # and choose the class with the largest probability.
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        self.params = [self.W, self.b]

        self.cost = cost or self.negative_log_likelihood

    def negative_log_likelihood(self, y):
        """
        A negative log likelihood cost function.

        :param y: set of true labels.
        :return: the negative log likelihood of y
        """
        return -T.mean(T.log(self.p_y_given_x[T.arange(y.shape[0]), y]))

    def errors(self, y):
        assert y.dtype.startswith('int')

        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 'y', y.type, 'y_pred', self.y_pred.type)

        return T.mean(T.neq(self.y_pred, y))
