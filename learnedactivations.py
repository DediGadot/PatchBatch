import numpy as np
import theano
import theano.tensor as T
from theano import ifelse
from lasagne import init
from lasagne import nonlinearities

from lasagne import layers


__all__ = [
    "BatchNormalizationLayer"
]


class BatchNormalizationLayer(layers.base.Layer):
    """
    Batch normalization Layer [1]
    The user is required to setup updates for the learned parameters (Gamma
    and Beta). The values nessesary for creating the updates can be
    obtained by passing a dict as the moving_avg_hooks keyword to
    get_output().

    REF:
     [1] http://arxiv.org/abs/1502.03167

    :parameters:
        - input_layer : `Layer` instance
            The layer from which this layer will obtain its input

        - nonlinearity : callable or None (default: lasagne.nonlinearities.rectify)
            The nonlinearity that is applied to the layer activations. If None
            is provided, the layer will be linear.

        - epsilon : scalar float. Stabilizing training. Setting this too
            close to zero will result in nans.

    :usage:
        >>> from lasagne.layers import InputLayer, BatchNormalizationLayer,
             DenseLayer
        >>> from lasagne.nonlinearities import linear, rectify
        >>> l_in = InputLayer((100, 20))
            l_dense = Denselayer(l_in, 50, nonlinearity=linear)
        >>> l_bn = BatchNormalizationLayer(l_dense, nonlinearity=rectify)
        >>> hooks, input, updates = {}, T.matrix, []
        >>> l_out = l_bn.get_output(
              input, deterministic=False, moving_avg_hooks=hooks)
        >>> mulfac = 1.0/100.0
        >>> batchnormparams = list(itertools.chain(
              *[i[1] for i in hooks['BatchNormalizationLayer:movingavg']]))
        >>> batchnormvalues = list(itertools.chain(
              *[i[0] for i in hooks['BatchNormalizationLayer:movingavg']]))
        >>> for tensor, param in zip(tensors, params):
                updates.append((param, (1.0-mulfac)*param + mulfac*tensor))
            # append updates to your normal update list
    """
    def __init__(self, incoming,
                 gamma = init.Uniform([0.95, 1.05]),
                 beta = init.Constant(0.),
                 nonlinearity=nonlinearities.rectify,
                 epsilon = 0.001,
                 **kwargs):
        super(BatchNormalizationLayer, self).__init__(incoming, **kwargs)
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.num_units = int(np.prod(self.input_shape[1:]))
        self.gamma = self.add_param(gamma, (self.num_units,),
                                   name="BatchNormalizationLayer:gamma",trainable=True)
        self.beta = self.add_param(beta, (self.num_units,),
                           name="BatchNormalizationLayer:beta",trainable=True)
        self.epsilon = epsilon

        self.mean_inference = theano.shared(
            np.zeros((1, self.num_units), dtype=theano.config.floatX),
            borrow=True,
            broadcastable=(True, False))
        self.mean_inference.name = "shared:mean-" + self.name ####

        self.variance_inference = theano.shared(
            np.zeros((1, self.num_units), dtype=theano.config.floatX),
            borrow=True,
            broadcastable=(True, False))
        self.variance_inference.name = "shared:variance-" + self.name ####

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, moving_avg_hooks=None,
                       deterministic=False, *args, **kwargs):
            
        reshape = False
        if input.ndim > 2:
            output_shape = input.shape
            reshape = True
            input = input.flatten(2)

        if deterministic is False:
            m  = T.mean(input, axis=0, keepdims=True)
            v = T.sqrt(T.var(input, axis=0, keepdims=True)+self.epsilon)
            m.name = "tensor:mean-" + self.name
            v.name = "tensor:variance-" + self.name

            key = "BatchNormalizationLayer:movingavg"
            if key not in moving_avg_hooks:
#                moving_avg_hooks[key] = {}
                moving_avg_hooks[key] = []
#            moving_avg_hooks[key][self.name] = [[m,v], [self.mean_inference, self.variance_inference]]
            moving_avg_hooks[key].append([[m,v], [self.mean_inference, self.variance_inference]])
        else:
            m = self.mean_inference
            v = self.variance_inference

        input_hat = (input - m) / v            # normalize
        y = self.gamma*input_hat + self.beta        # scale and shift

        if reshape:#input.ndim > 2:
            y = T.reshape(y, output_shape)
        return self.nonlinearity(y)
