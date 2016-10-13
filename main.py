import lasagne
import numpy
import pb_NN as NN
import pb_Models as Models
import cv2
from sklearn.feature_extraction import image
import pb_utils as utils
import patchmodule
import kittitool
import matplotlib.pyplot as pyplot

DEBUG = True

patch_size = 51
batch_size = 256
# pm params suitable for kitti2012
pm_params = (2, 5, 10, 500) # num iters, random search iters, max_h, max_w
pm_iters, pm_rs, rand_ann_h, rand_ann_w = pm_params


def calc_descs(img1_filename, img2_filename, model_name):
    net_name, weights_filename, eparams_filename = Models.nets[model_name]
    nn_model, theano_func = NN.get_net_and_funcs(net_name, batch_size, weights_filename, eparams_filename)

    img_descs = []
    for img_filename in [img1_filename, img2_filename]:
        print 'reading', img_filename
        img = cv2.imread(img_filename, 0)
        #if DEBUG:
        #    img = cv2.resize(img, (img.shape[1]/4, img.shape[0]/4))
        h,w = img.shape
        print 'image shape before reflect', img.shape

        # reflect borders
        img = cv2.copyMakeBorder(img,patch_size/2,patch_size/2,patch_size/2,patch_size/2,cv2.BORDER_REFLECT_101)
        print 'image shape after reshape', img.shape

        # extract patches and reshape
        patches = image.extract_patches_2d(img, (patch_size, patch_size))
        patches = patches.reshape(h, w, patch_size, patch_size)
        print 'patches shape', patches.shape

        # create descriptors
        print 'starting to create descriptors with patch_size', patch_size, 'and batch_size', batch_size
        descs = NN.get_descriptors(nn_model, theano_func, patches, batch_size, patch_size)

        img_descs.append(descs)

    return img_descs

def calc_patchmatch_and_cost(img1_descs, img2_descs, pm_params, both = False):
    """ returns transformed flows """
    h,w = img1_descs.shape[:2]

    print 'calculating patchmatch A->B'
    rand_ann_h, rand_ann_w = pm_params[2:4]
    rand_ann = utils.create_random_ann(h, w, rand_ann_h, rand_ann_w)
    ann_AB, matchcost_AB = patchmodule.patchmatch(img1_descs, img2_descs, numpy.copy(rand_ann),pm_iters = pm_iters, rs_start = pm_rs)
    annAB_trans = utils.transform_flow(ann_AB)
    annAB_trans = numpy.concatenate([annAB_trans, numpy.ones([annAB_trans.shape[0], annAB_trans.shape[1], 1], dtype=numpy.float32)], axis=2)

    if both:
        print 'calculating patchmatch B->A'
        ann_BA, matchcost_BA = patchmodule.patchmatch(img2_descs, img1_descs, numpy.copy(rand_ann),pm_iters = pm_iters, rs_start = pm_rs)
        annBA_trans = utils.transform_flow(ann_BA)
        annBA_trans = numpy.concatenate([annBA_trans, numpy.ones([annBA_trans.shape[0], annBA_trans.shape[1], 1], dtype=numpy.float32)], axis=2)
        res = {'flow' : [annAB_trans, annBA_trans],
                'cost' : [matchcost_AB, matchcost_BA]}

    else:
        res = {'flow' : [annAB_trans],
               'cost' : [matchcost_AB]}

    return res


def calc_flow_and_cost(img1_filename, img2_filename, net_name, eliminate_bidi_errors = False):
    img1_descs, img2_descs = calc_descs(img1_filename, img2_filename, net_name)

    flows_costs = calc_patchmatch_and_cost(img1_descs, img2_descs, pm_params, eliminate_bidi_errors)
    flows = flows_costs['flow']
    costs = flows_costs['cost']

    if eliminate_bidi_errors:
        flowAB, flowBA = flows
        bidi_errors = utils.calc_bidi_errormap(flowAB[...,:2], flowBA[...,:2])
        flow_res = flows[0]
        flow_res[bidi_errors, :] = 0
        cost_res = costs[0]
        cost_res[bidi_errors] = 0

    else:
        flow_res = flows[0]
        cost_res = costs[0]

    return [flow_res, cost_res]



if __name__ == '__main__':

    model_name = 'KITTI2012_CENTSD_ACCURATE'
    img1_filename = '/home/MAGICLEAP/dgadot/patchflow_data/training/image_0/000000_10.png'
    img2_filename = '/home/MAGICLEAP/dgadot/patchflow_data/training/image_0/000000_11.png'

    flow_res, cost_res = calc_flow_and_cost(img1_filename, img2_filename, model_name, eliminate_bidi_errors = True)
    kittitool.flow_visualize(flow_res, mode='Y')
