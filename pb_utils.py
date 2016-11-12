import numpy
import matplotlib.pyplot as pyplot
import cv2

def benchmark_flow(flow, gt_flow, debug=False, tau=3):
    valid_inds = gt_flow[:,:,2] == 1
    valid_flow = flow[valid_inds, :2]
    valid_gt = gt_flow[valid_inds, :2]

    euc_err = numpy.sqrt(numpy.sum((valid_flow-valid_gt)**2, axis=1))

    perc_valid = float(valid_gt.shape[0]) / (gt_flow.shape[0] * gt_flow.shape[1])
    avg_err = numpy.mean(euc_err)
    perc_above_tau = float(numpy.sum(euc_err > tau)) / valid_gt.shape[0]

    print 'avg_err %.2f perc_above_tau %.2f% perc_valid %.2f%' % (avg_err, perc_above_tau*100, perc_valid*100)

    if debug:
        pyplot.figure()

        pyplot.subplot(2,2,1)
        pyplot.imshow(flow_to_color(flow[:,:,:2]))
        pyplot.subplot(2,2,2)
        pyplot.imshow(flow_to_color(gt_flow[:,:,:2]))

        pyplot.subplot(2,1,2)
        tau_map = numpy.sum((flow[:,:,:2] - gt_flow[:,:,:2])**2, axis=2)
        tau_map = tau_map > tau
        non_valid_inds = gt_flow[:,:,2] != 1
        tau_map[non_valid_inds] = 0
        pyplot.imshow(tau_map)
        myshow()

def disp_flow(flow,title=None):
    pyplot.figure()
    if 'int' in str(flow.dtype):
        flow = transform_flow(flow)
    pyplot.imshow(flow_to_color(flow))
    if title is not None: pyplot.title(title)

    # just for visualization purposes, eliminate top 10% of flow
    #tmp = flow[:,:,0]**2 + flow[:,:,1]**2
    #flow[tmp > numpy.percentile(tmp, 90)] = 0

    myshow()

def myshow():
    figManager = pyplot.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    pyplot.tight_layout()
    pyplot.show()
    pyplot.pause(0.5)

def mydisp(img, title=None):
    pyplot.imshow(img)
    if title is not None:
        pyplot.title(title)
    pyplot.show()
    pyplot.pause(0.5)

def transform_flow(flow):
    h,w = flow.shape[0:2]
    x_mat = (numpy.expand_dims(range(w),0) * numpy.ones((h,1),dtype=numpy.int32)).astype(numpy.int32)
    y_mat = (numpy.ones((1,w),dtype=numpy.int32) * numpy.expand_dims(range(h),1)).astype(numpy.int32)

    res = numpy.copy(flow)
    res[:,:,0] = res[:,:,0] - x_mat
    res[:,:,1] = res[:,:,1] - y_mat

    return res.astype(numpy.float32)

def create_random_ann(h, w, max_h, max_w):

    w_mat = (numpy.expand_dims(range(w),0) * numpy.ones((h,1),dtype=numpy.int32)).astype(numpy.int32)
    h_mat = (numpy.ones((1,w),dtype=numpy.int32) * numpy.expand_dims(range(h),1)).astype(numpy.int32)
    rand_h = numpy.random.choice(range(-1 * max_h,max_h + 1),(h,w))
    rand_w = numpy.random.choice(range(-1 * max_w,max_w + 1),(h,w))

    ann = numpy.zeros((h,w,2),dtype=numpy.int32)
    ann[...,0] = w_mat + rand_w
    ann[...,0][ann[...,0] < 0] = 0
    ann[...,0][ann[...,0] > w-1] = w-1
    ann[...,1] = h_mat + rand_h
    ann[...,1][ann[...,1] < 0] = 0
    ann[...,1][ann[...,1] > h-1] = h-1

    return ann

def calc_bidi_errormap(flowAB,flowBA,tau=1):
    """ transformed flows as input """
    h,w = flowAB.shape[0:2]
    x_mat = (numpy.expand_dims(range(w),0) * numpy.ones((h,1),dtype=numpy.int32)).astype(numpy.int32)
    y_mat = (numpy.ones((1,w),dtype=numpy.int32) * numpy.expand_dims(range(h),1)).astype(numpy.int32)

    d1 = flowAB
    r_cords = (y_mat + d1[:,:,1]).astype(numpy.int32)
    c_cords = (x_mat + d1[:,:,0]).astype(numpy.int32)
    r_cords[r_cords>h-1] = h-1
    r_cords[r_cords<0] = 0
    c_cords[c_cords>w-1] = w-1
    c_cords[c_cords<0] = 0

    d2 = flowBA[r_cords,c_cords,:]
    d = numpy.sqrt(numpy.sum((d1+d2)**2,axis=2))

    bidi_map = d > tau

    return bidi_map

def flow_to_color(flow):
    hsv = numpy.zeros((flow.shape[0],flow.shape[1],3),dtype=numpy.uint8)
    hsv[...,1] = 255

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/numpy.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    return rgb
