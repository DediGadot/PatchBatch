import numpy

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
