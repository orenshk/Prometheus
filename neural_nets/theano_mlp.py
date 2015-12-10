import theano as th
import theano.tensor as T
import numpy as np
from theano_logistic_regression import LogisticRegression


class HiddenLayer(object):
    def __init__(self, input_data, n_in, n_out, W=None, b=None, activation=T.tanh):
        """
       Typical hidden layer of a MLP: units are fully-connected and have
       sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
       and the bias vector b is of shape (n_out,).

       NOTE : The nonlinearity used here is tanh

       Hidden unit activation is given by: tanh(dot(input,W) + b)

       :type input_data: theano.tensor.dmatrix
       :param input_data: a symbolic tensor of shape (n_examples, n_in)

       :type n_in: int
       :param n_in: dimensionality of input

       :type n_out: int
       :param n_out: number of hidden units

       :type activation: theano.Op or function
       :param activation: Non linearity to be applied in the hidden
                          layer
       """

        rng = np.random.RandomState(1234)

        self.input = input_data

        self.W = W or self._init_weights(rng, n_in, n_out, activation)
        self.b = b or self._init_bias(n_out)

        activation = activation or (lambda x: x)
        self.output = activation(T.dot(self.input, self.W) + self.b)

        self.params = [self.W, self.b]

    @staticmethod
    def _init_weights(rng, n_in, n_out, activation):
        """
        `W` is initialized with `W_values` which is uniformely sampled
        from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        for tanh activation function
        the output of uniform if converted using asarray to dtype
        theano.config.floatX so that the code is runable on GPU
        Note : optimal initialization of weights is dependent on the
               activation function used (among other things).
               For example, results presented in [Xavier10] suggest that you
               should use 4 times larger initial weights for sigmoid
               compared to tanh
               We have no info for other function, so we use the same as
               tanh.
        """

        degree = n_in + n_out
        W_values = np.asarray(
            rng.uniform(
                low=-np.sqrt(6. / degree),
                high=np.sqrt(6. / degree),
                size=(n_in, n_out)
            ),
            dtype=th.config.floatX
        )
        if activation == T.nnet.sigmoid:
            W_values *= 4

        return th.shared(value=W_values, name='W', borrow=True)

    @staticmethod
    def _init_bias(n_out):
        b_values = np.zeros((n_out,), dtype=th.config.floatX)
        return th.shared(value=b_values, name='b', borrow=True)


class MultilayerPerceptron(object):
    def __init__(self, input_data, n_in, n_hidden, n_out, L1_reg=0., L2_reg=0.0001, cost='neg_log'):
        """
        Initialize the parameters for the multilayer perceptron

        :type input_data: theano.tensor.TensorType
        :param input_data: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie
        """

        self.hidden_layer = HiddenLayer(
            input_data=input_data,
            n_in=n_in,
            n_out=n_hidden,
            activation=T.tanh
        )

        if cost == 'neg_log':
            # this will cause log_regression_layer to choose negative log likelihood
            cost_func = None
        elif cost == 'cross_ent':
            cost_func = self.cross_entropy
        else:
            raise NotImplementedError('cost function {} not implemented'.format(cost))

        self.log_regression_layer = LogisticRegression(
            input_data=self.hidden_layer.output,
            n_in=n_hidden,
            n_out=n_out,
            cost=cost_func
        )

        self.L1 = abs(self.hidden_layer.W).sum() + abs(self.log_regression_layer.W).sum()
        self.L2 = abs(self.hidden_layer.W ** 2).sum() + abs(self.log_regression_layer.W ** 2).sum()

        self.cost = self.log_regression_layer.cost
        self.errors = self.log_regression_layer.errors
        self.params = self.hidden_layer.params + self.log_regression_layer.params

        self.input = input_data

    def cross_entropy(self, y):
        clf = self.log_regression_layer
        return -T.sum(y * T.log(clf.p_y_given_x[T.arange(y.shape[0]), y]))
