import lasagne
import theano
from lasagne import nonlinearities
import lasagne.layers as layers
from learnedactivations import BatchNormalizationLayer
import os

leaky_param = 0.1
in_channels = 1
border_mode = 'valid'

cur_dir = os.path.dirname(os.path.realpath(__file__))

# <net_name> : [Lasagne model, weights_filename, batchnorm_weights_filename<]
nets = {'KITTI2015_CENTSD_ACCURATE' : ['model_CENTSD_33conv',
                              cur_dir + '/weights/KITTI2015_ACCURATE/211015_141323PAPERKITTI2015-model_drlim7_33conv_allconv_neg1_8_m100_epoch4000_adadelta_testsamples800k_impdrlimv3.yaml-best-test-weights.pickle',
                              cur_dir + '/weights/KITTI2015_ACCURATE/211015_141323-eparams-test.pickle'],

        'KITTI2012_CENTSD_ACCURATE' : ['model_CENTSD_33conv',
                                       cur_dir + '/weights/KITTI2012_ACCURATE/241015_080511PAPERKITTI2012-model_drlim7_33conv_allconv_neg1_8_m100_epoch4000_adadelta_testsamples800k_impdrlimv3.yaml-best-test-weights.pickle',
                                       cur_dir + '/weights/KITTI2012_ACCURATE/241015_080511-eparams-test.pickle'],

        'KITTI2015_SPCI' : ['model_CENTSD_33conv_elu',
                                       cur_dir + '/weights/KITTI2015_SPCI/111016_151719PAPERKITTI2015-model_drlim7_33conv_elu_allconv_neg1_8_m100_epoch4000_adadelta_testsamples200k_hingelosssd_p71_normsamp_sp_load_k15.yaml-best-test-weights.pickle',
                                       cur_dir + '/weights/KITTI2015_SPCI/111016_151719-eparams-test.pickle'],

        'KITTI2012_SPCI' : ['model_CENTSD_33conv_elu',
                                       cur_dir + '/weights/KITTI2012_SPCI/131016_180941PAPERKITTI2012-model_drlim7_33conv_elu_allconv_neg1_8_m100_epoch4000_adadelta_testsamples200k_hingelosssd_p71_normsamp_sp_load.yaml-best-test-weights.pickle',
                                       cur_dir + '/weights/KITTI2012_SPCI/131016_180941-eparams-test.pickle']}

def layer_factory(in_layer, layer_type, **kwargs):
    """ Given an input layer and parameters, creates a Lasagen layer """

    gpu = True if 'gpu' in theano.config.device else False
    if gpu:
        from lasagne.layers import dnn
    if layer_type == 'conv':
        func = dnn.Conv2DDNNLayer if gpu else layers.Conv2DLayer
        defaults = {'border_mode':'same','W':lasagne.init.GlorotUniform()}  ### dimshuffle=TRUE!!!
    elif layer_type == 'dense':
        func = layers.DenseLayer
        defaults = {'W':lasagne.init.Uniform()}
    elif layer_type == 'maxout':
        defaults = {}
        func = dnn.MaxPool2DDNNLayer if gpu else layers.MaxPool2DLayer
    else:
        return -1

    layer_params = {}
    for key,val in kwargs.iteritems():
        if layer_type in key:
            new_key = key.split(layer_type)[1]
            new_key = new_key[1:]
            layer_params[new_key] = val

    if layer_type == 'maxout':
        in_layers = []
        for i in xrange(layer_params['K']):
            tmp_layer = layers.DenseLayer(in_layer,num_units=layer_params['num_units'],W=lasagne.init.GlorotUniform(),nonlinearity=lasagne.nonlinearities.linear)
            tmp_layer = layers.ReshapeLayer(tmp_layer,([0],1,[1]))
            in_layers.append(tmp_layer)
        in_layer = lasagne.layers.ConcatLayer(tuple(in_layers))
        orig_nonlin = lasagne.nonlinearities.identity
        layer_params.pop('K',None)
        layer_params.pop('num_units',None)

    name = kwargs['name'] + '_' if 'name' in kwargs else 'NONE_'

    if 'batch_norm' in kwargs and kwargs['batch_norm'] and layer_type != 'maxout':
        orig_nonlin = layer_params['nonlinearity']
        layer_params['nonlinearity'] = lasagne.nonlinearities.linear

    # remove user-configurations from defaults
    for key in layer_params.keys():
        if key in defaults:
            defaults.pop(key,None)

    all_params = dict(defaults,**layer_params)

    # new lasagne!
    if 'border_mode' in all_params:
        all_params['pad'] = all_params['border_mode']
        del all_params['border_mode']

    output_layer = func(in_layer,**all_params)

    if 'batch_norm_f0k' in kwargs and kwargs['batch_norm_f0k']:
        output_layer = lasagne.layers.normalization.batch_norm(output_layer)

    if 'batch_norm' in kwargs and kwargs['batch_norm']:
            output_layer = BatchNormalizationLayer(output_layer,nonlinearity=orig_nonlin,name=name + 'batch')

    if 'maxpool' in kwargs and kwargs['maxpool']:
        st = kwargs['maxpool_st'] if 'maxpool_st' in kwargs else None
        ignore_borders = kwargs['ignore_borders'] if 'ignore_borders' in kwargs else False
        output_layer = lasagne.layers.MaxPool2DLayer(output_layer,pool_size=kwargs['maxpool_ds'],stride=st,name=name + 'maxpool',ignore_border=ignore_borders)

    return output_layer

def model_CENTSD_33conv(patch_size, batch_size, FAST_network=False, FAST_imgheight=None, FAST_imgwidth=None, nonlin_func='leaky'):
    """ Describes the main network used in the PatchBatch paper """

    if nonlin_func == 'leaky':
        nonlin = nonlinearities.LeakyRectify(leaky_param)
    elif nonlin_func == 'elu':
        nonlin = nonlinearities.elu
    else:
        print 'Error! Unsupported non-linearity function'
        return

    if FAST_network:
        l_in0 = layers.InputLayer(
            shape=(1, 1, FAST_imgheight, FAST_imgwidth),name='l_in0')
    else:
        l_in0 = layers.InputLayer(
            shape=(batch_size, in_channels, patch_size, patch_size),name='l_in0')

    layer_params = {'conv_num_filters':32,
                    'conv_filter_size':(3,3),
                    'conv_border_mode':border_mode,
                    'conv_nonlinearity':nonlin,
                    'batch_norm':True,
                    'maxpool':True,
                    'maxpool_ds':(2,2)}
    layer = layer_factory(in_layer=l_in0,layer_type='conv',**layer_params)

    layer_params = {'conv_num_filters':64,
                    'conv_filter_size':(3,3),
                    'conv_border_mode':border_mode,
                    'conv_nonlinearity':nonlin,
                    'batch_norm':True,
                    'maxpool':True,
                    'maxpool_ds':(2,2)}
    layer = layer_factory(in_layer=layer,layer_type='conv',**layer_params)

    layer_params = {'conv_num_filters':128,
                    'conv_filter_size':(3,3),
                    'conv_border_mode':border_mode,
                    'conv_nonlinearity':nonlin,
                    'batch_norm':True,
                    'maxpool':True,
                    'maxpool_ds':(2,2)}
    layer = layer_factory(in_layer=layer,layer_type='conv',**layer_params)

    layer_params = {'conv_num_filters':256,
                    'conv_filter_size':(3,3),
                    'conv_border_mode':border_mode,
                    'conv_nonlinearity':nonlin,
                    'batch_norm':True,
                    'maxpool':True,
                    'maxpool_ds':(2,2)}
    layer = layer_factory(in_layer=layer,layer_type='conv',**layer_params)

    layer_params = {'conv_num_filters':512,
                    'conv_filter_size':(2,2),
                    'conv_border_mode':border_mode,
                    'conv_nonlinearity':nonlin,
                    'batch_norm':True,
                    'maxpool':False,
                    'maxpool_ds':(2,2)}
    layer = layer_factory(in_layer=layer,layer_type='conv',**layer_params)

    return layer


def model_CENTSD_33conv_elu(batch_size,FAST_network=False, FAST_imgheight=None, FAST_imgwidth=None):
    """ creates a CENTSD_33conv model with elu nonlinearity """

    return model_CENTSD_33conv(batch_size, FAST_network, FAST_imgheight, FAST_imgwidth, nonlin_func = 'elu')


all_models = {'model_CENTSD_33conv' : model_CENTSD_33conv,
              'model_CENTSD_33conv_elu' : model_CENTSD_33conv_elu}
