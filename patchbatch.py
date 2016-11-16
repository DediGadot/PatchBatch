import lasagne
import sys
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

DEBUG = False

def calc_descs(img1_filename, img2_filename, model_name, patch_size, batch_size):
    """ given two image files and a CNN model name, calculates the dense descriptor tensor of both images
        img1_filename - full path of source image
        img2_filename - full path of target image
        model_name - name of the trained CNN to use, out of the supported models, documented in pb_Models """

    net_name, weights_filename, eparams_filename = Models.nets[model_name]
    nn_model, theano_func = NN.get_net_and_funcs(net_name, patch_size, batch_size, weights_filename, eparams_filename)

    img_descs = []
    for img_filename in [img1_filename, img2_filename]:
        print 'reading', img_filename
        img = cv2.imread(img_filename, 0)
        if DEBUG:
            img = cv2.resize(img, (img.shape[1]/4, img.shape[0]/4))
        h,w = img.shape
        # normalize image
        img = ((img.astype(numpy.float32) - numpy.mean(img)) / numpy.std(img))
        print 'img normalized, mean %.2f std %.2f' % (numpy.mean(img), numpy.std(img))

        # reflect borders
        print 'image shape before reflect', img.shape
        img = cv2.copyMakeBorder(img,patch_size/2,patch_size/2,patch_size/2,patch_size/2,cv2.BORDER_REFLECT_101)
        print 'image shape after reflect', img.shape


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
        pm_params - <x,y,z,w>:
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
    pm_iters, pm_rs, rand_ann_h, rand_ann_w = pm_params
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


def calc_flow_and_cost(img1_descs, img2_descs, pm_params, eliminate_bidi_errors = False):
    """ Given two descriptor tensors, calculate PatchMatch and return flow + cost
        img1_descs - <h,w,1,#d> descriptor tensor for the source image
        img2_descs - <h,w,1,#d> descriptor tensor for the target image
        pm_params - <x,y,z,w>:
            x - number of PatchMatch iters to do
            y - number of random searches to do, as part of PatchMatch
            z - max h value for initial random ANN
            w - max w value for initial random ANN
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

def calc_flow(img1_filename, img2_filename, model_name, output_filename, patch_size=51, batch_size=256, bidi=False):
    """ Given two input filenames, model_name and output_filename return+save flow res
        img1_filename - filename of source image
        img2_filename - filename of target image
        model_name - name of the trained CNN to use, out of the supported models documented in pb_Models
        output_filename - filename of output pickle file, containing either flowAB or flowAB and flowBA depennding on bidi flag
        bidi - whether to compute flowAB and flowBA and do a bidirectional consistency check

        Returns:
            (flowAB) or (flowAB and flowBA) depending on bidi flag """

    if 'KITTI' in model_name:
        pm_params = (2, 5, 10, 500) # pm_iters, pm_random_search_iters, rand_max_h, rand_max_w
    elif 'MPI' in model_name:
        pm_params = (2, 20, 10, 10)

    print 'Calculating descriptors...'
    img_descs = calc_descs(img1_filename, img2_filename, model_name, patch_size, batch_size)
    print 'Calculating flow fields and matching cost'
    flow_res, cost_res = calc_flow_and_cost(img_descs[0], img_descs[1], pm_params, bidi)

    if output_filename is not None:
        print 'Saving flow to', output_filename
        with open(output_filename, 'wb') as f:
            pickle.dump(flow_res, f)

    print 'flow coverage percentage: %.2f' % (numpy.sum(flow_res[:,:,2]) / (flow_res.shape[0] * flow_res.shape[1]))

    return flow_res


def main(patch_size=51, batch_size=256):
    parser = argparse.ArgumentParser(description = 'PatchBatch Optical Flow algorithm')
    parser.add_argument('img1_filename', help = 'Filename (+path) of the source image')
    parser.add_argument('img2_filename', help = 'Filename (+path) of the target image')
    parser.add_argument('model_name', help = 'Name of network to run')
    parser.add_argument('output_path', help = 'Path to where to place the results')
    parser.add_argument('--bidi', help  = 'Run bidirectional consistency test, mark invalid correspondences as such', action='store_true')
    parser.add_argument('--descs', help  = 'Save descriptors to file', action='store_true')
    parser.add_argument('--debug', help  = 'Downscales input images by a factor of 4 for debugging purposes', action='store_true')

    parser = parser.parse_args()

    if not os.path.exists(parser.output_path):
        os.mkdir(parser.output_path)

    if 'KITTI' in parser.model_name:
        pm_params = (2, 5, 10, 500) # pm_iters, pm_random_search_iters, rand_max_h, rand_max_w
    elif 'MPI' in parser.model_name:
        pm_params = (2, 20, 10, 10)
    else:
        print 'Error! Unsupported pm_params'
        sys.exit()

    if 'SPCI' in parser.model_name:
        patch_size = 71
        batch_size = 255

    #model_name = 'KITTI2012_CENTSD_ACCURATE'
    #img1_filename = '/home/MAGICLEAP/dgadot/patchflow_data/training/image_0/000000_10.png'
    #img2_filename = '/home/MAGICLEAP/dgadot/patchflow_data/training/image_0/000000_11.png'
    #output_path = '/tmp'

    #img_descs = calc_descs(img1_filename, img2_filename, model_name)
    #flow_res, cost_res = calc_flow_and_cost(img_descs[0], img_descs[1], True)

    if parser.debug:
        DEBUG = True
        print 'DEBUG mode is', DEBUG

    print 'Calculating descriptors...'
    img_descs = calc_descs(parser.img1_filename, parser.img2_filename, parser.model_name, patch_size, batch_size)
    print 'Calculating flow fields and matching cost'
    flow_res, cost_res = calc_flow_and_cost(img_descs[0], img_descs[1], pm_params, parser.bidi)

    print 'flow coverage percentage: %.2f' % (numpy.sum(flow_res[:,:,2]) / (flow_res.shape[0] * flow_res.shape[1]))

    print 'Saving outputs to', parser.output_path
    with open(parser.output_path + '/flow.pickle','wb') as f:
        pickle.dump(flow_res, f)

    with open(parser.output_path + '/cost.pickle','wb') as f:
        pickle.dump(cost_res, f)

    if parser.descs:
        with open(parser.output_path + '/descs.pickle', 'wb') as f:
            pickle.dump(img_descs, f)

    kittitool.flow_visualize(flow_res, mode='Y')


if __name__ == '__main__':
    main()
