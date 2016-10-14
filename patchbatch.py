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
import argparse
import os
import cPickle as pickle

DEBUG = True

patch_size = 51
batch_size = 256
# pm params suitable for kitti2012
pm_params = (2, 5, 10, 500) # num iters, random search iters, max_h, max_w
pm_iters, pm_rs, rand_ann_h, rand_ann_w = pm_params


def calc_descs(img1_filename, img2_filename, model_name):
    """ given two image files and a CNN model name, calculates the dense descriptor tensor of both images
        img1_filename - full path of source image
        img2_filename - full path of target image
        model_name - name of the trained CNN to use, out of the supported models """

    net_name, weights_filename, eparams_filename = Models.nets[model_name]
    nn_model, theano_func = NN.get_net_and_funcs(net_name, batch_size, weights_filename, eparams_filename)

    img_descs = []
    for img_filename in [img1_filename, img2_filename]:
        print 'reading', img_filename
        img = cv2.imread(img_filename, 0)
        if DEBUG:
            img = cv2.resize(img, (img.shape[1]/4, img.shape[0]/4))
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
    """ Given two descriptor tensors, and PatchMatch params, calculate the ANN
        img1_descs - <h,w,1,#d> dense descriptor tensor for source image
        img2_descs - <hw,w,1,#d> dense descriptor tensor for target image
        pm_params - <x,y,z,w> (see default values above):
            x - number of PatchMatch iters to do
            y - number of random searches to do, as part of PatchMatch
            z - max h value for initial random ANN
            w - max w value for initial random ANN

        Note: pm_params parameters were empirically identified using a training
        set on KITTI2012, KITTI2015, MPI-Sintel respectively

        Returns:
            A dictionary with either one or two flow tensors and either one or
            two cost tensors, depending on the both argument
            Flow tensors are <h,w,3>, with the channels being: U,V,valid? """


    h,w = img1_descs.shape[:2]

    print 'calculating patchmatch A->B'
    rand_ann_h, rand_ann_w = pm_params[2:4]
    rand_ann = utils.create_random_ann(h, w, rand_ann_h, rand_ann_w)
    ann_AB, matchcost_AB = patchmodule.patchmatch(img1_descs, img2_descs, numpy.copy(rand_ann),pm_iters = pm_iters, rs_start = pm_rs)

    # transform OF result from image-pixel coordinates to relative pixel values
    annAB_trans = utils.transform_flow(ann_AB)
    # add a valid channel to the OF tensor
    annAB_trans = numpy.concatenate([annAB_trans, numpy.ones([annAB_trans.shape[0], annAB_trans.shape[1], 1], dtype=numpy.float32)], axis=2)

    # if both: calculate also B->A patchmatch
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


def calc_flow_and_cost(img1_descs, img2_descs, eliminate_bidi_errors = False):
    """ Given two descriptor tensors, calculate PatchMatch and return flow + cost
        img1_descs - <h,w,1,#d> descriptor tensor for the source image
        img2_descs - <h,w,1,#d> descriptor tensor for the target image
        eliminate_bidi_errors - if True, mark as invalid correspondences which
        do not meet the bidirectional consistency check

        Returns:
            optical flow from source to target, with <U,V,valid?> channels
            cost tensor describing the matching cost """

    flows_costs = calc_patchmatch_and_cost(img1_descs, img2_descs, pm_params, both = eliminate_bidi_errors)
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

    return flow_res, cost_res

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'PatchBatch Optical Flow algorithm')
    parser.add_argument('img1_filename', help = 'Filename (+path) of the source image')
    parser.add_argument('img2_filename', help = 'Filename (+path) of the target image')
    parser.add_argument('model_name', help = 'Name of network to run')
    parser.add_argument('output_path', help = 'Path to where to place the results')
    parser.add_argument('--bidi', help  = 'Run bidirectional consistency test, mark invalid correspondences as such', action='store_true')

    parser = parser.parse_args()

    if not os.path.exists(parser.output_path):
        os.mkdir(parser.output_path)


    #model_name = 'KITTI2012_CENTSD_ACCURATE'
    #img1_filename = '/home/MAGICLEAP/dgadot/patchflow_data/training/image_0/000000_10.png'
    #img2_filename = '/home/MAGICLEAP/dgadot/patchflow_data/training/image_0/000000_11.png'
    #output_path = '/tmp'

    #img_descs = calc_descs(img1_filename, img2_filename, model_name)
    #flow_res, cost_res = calc_flow_and_cost(img_descs[0], img_descs[1], True)

    print 'Calculating descriptors...'
    img_descs = calc_descs(parser.img1_filename, parser.img2_filename, parser.model_name)
    print 'Calculating flow fields and matching cost'
    flow_res, cost_res = calc_flow_and_cost(img_descs[0], img_descs[1], parser.bidi)

    print 'Saving outputs to', parser.output_path
    with open(parser.output_path + '/flow.pickle','wb') as f:
        pickle.dump(flow_res, f)

    with open(parser.output_path + '/cost.pickle','wb') as f:
        pickle.dump(cost_res, f)

    with open(parser.output_path + '/descs.pickle', 'wb') as f:
        pickle.dump(img_descs, f)

    #kittitool.flow_visualize(flow_res, mode='Y')
