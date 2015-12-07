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

        self.params = [self.W, self.b]

        self.cost = cost or self.negative_log_likelihood

    def predict(self, x):
        return T.argmax(self.p_of_y_given(x), axis=1)

    def p_of_y_given(self, x):
        return T.nnet.softmax(T.dot(x, self.W) + self.b)

    def negative_log_likelihood(self, x, y):
        """
        A negative log likelihood cost function.

        :param y: set of true labels.
        :return: the negative log likelihood of y
        """
        return -T.mean(T.log(self.p_of_y_given(x)[T.arange(y.shape[0]), y]))

    def adv_cost(self, x, y):
        alpha = 0.5
        eps = 0.1

        p_y_x = T.nnet.softmax(T.dot(x, self.W) + self.b)
        J = -T.mean(T.log(p_y_x[T.arange(y.shape[0]), y]))
        grad_J = T.grad(cost=J, wrt=x)

        self.x_hat = x + eps * grad_J / abs(grad_J)
        p_y_x_hat = T.nnet.softmax(T.dot(self.x_hat, self.W) + self.b)
        J_hat = -T.mean(T.log(p_y_x_hat[T.arange(y.shape[0]), y]))
        return alpha * J + (1 - alpha) * J_hat

    def errors(self, x, y):
        assert y.dtype.startswith('int')

        pred = self.predict(x)

        if y.ndim != pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', 'y', y.type, 'y_pred', self.y_pred.type)

        return T.mean(T.neq(pred, y))
