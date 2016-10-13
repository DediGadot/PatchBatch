import cPickle as pickle
import pb_Models as Models
import lasagne
import theano
import numpy
import os
from learnedactivations import BatchNormalizationLayer

cur_dir = os.path.dirname(os.path.realpath(__file__))

def set_batchnorm_params(nn_model,eparams_filename):
    with open(eparams_filename,'rb') as f:
        params = pickle.load(f)
    ind = 0
    for layer in lasagne.layers.get_all_layers(nn_model):
        if isinstance(layer,BatchNormalizationLayer):
            (layer.mean_inference).set_value(params[ind])
            (layer.variance_inference).set_value(params[ind+1])
            ind = ind + 2


def describe_network(output_layer):
    all_layers = lasagne.layers.get_all_layers(output_layer)
    weights = lasagne.layers.get_all_param_values(output_layer)
    weights = weights[0::2]
    ind = 0
    for layer in all_layers:
        ss = str(type(layer))
        ss = ss.split(' ')[1]
        ss = ss.replace("'",'')
        ss = ss.replace(">",'')
        ss = ss.split('.')[-1]
        tot = "ind" + ' ' + str(ind) + ' ' + ss + ' ' + str(lasagne.layers.get_output_shape(layer))
        if hasattr(layer,'nonlinearity'):
            nonlin = str(layer.nonlinearity)
            if 'object' in nonlin:
                nonlin = nonlin.split(' ')[0].split('<lasagne.nonlinearities.')[1]
            else:
                nonlin = nonlin.split(' ')[1]
            tot = tot + ' ' + nonlin
        cur_params = layer.get_params()
        if len(cur_params) != 0 and len(weights) != 0:
            tot = tot + ' ws=' + str(weights[0].shape)
            weights.pop(0)
        print tot
        ind = ind+1

def get_descriptors(nn_model, theano_func, patches, batch_size, patch_size):
    h,w = patches.shape[:2]
    patches = patches.reshape(-1, 1, patch_size, patch_size)
    num_batches = patches.shape[0] / batch_size
    descs = []
    for b in xrange(num_batches+1):
        if b % 200 == 0:
            print 'finished processing', b,' batches out of', num_batches

        min_slice = b* batch_size
        max_slice = min((b+1) * batch_size, patches.shape[0])
        batch_slice = slice(min_slice, max_slice)

        cur_patches = patches[batch_slice]
        cur_descs = theano_func(cur_patches)[0]
        descs.append(cur_descs.squeeze())

    res = numpy.vstack(descs)
    desc_size = res.shape[-1]
    res = res.reshape(h, w, 1, desc_size)

    return res


def get_net_and_funcs(net_name, batch_size, weights_filename, eparams_filename):
    print 'Creating NN', net_name
    nn_model = Models.all_models[net_name](batch_size)

    print 'Describing network:'
    describe_network(nn_model)

    print 'Loading and setting weights from', weights_filename
    with open(weights_filename, 'rb') as f:
        weights = pickle.load(f)
    lasagne.layers.set_all_param_values(nn_model, weights)

    print 'Loading and setting batch_norm params from', eparams_filename
    set_batchnorm_params(nn_model, eparams_filename)

    Xb = theano.tensor.tensor4('x')
    NN_output = lasagne.layers.get_output(nn_model, Xb, deterministic=True)

    theano_func = theano.function(
        inputs = [Xb],
        outputs = [NN_output],
    )

    return nn_model, theano_func




