# cython: profile=False

from __future__ import division
import numpy
cimport numpy
cimport cython

DTYPE = numpy.int32
DTYPE_F = numpy.float32
ctypedef numpy.int32_t DTYPE_t
ctypedef numpy.float32_t DTYPE_F_t

cdef inline int int_max(int a, int b): return a if a >= b else b
cdef inline int int_min(int a, int b): return a if a <= b else b
cdef inline float float_max(float a, float b): return a if a >= b else b
cdef inline float float_min(float a, float b): return a if a <= b else b

@cython.boundscheck(False)
cdef float dist(numpy.ndarray[DTYPE_F_t,ndim=4] imgA_descs,numpy.ndarray[DTYPE_F_t,ndim=4] imgB_descs,int yA,int xA,int yB,int xB, float dbest):
 
    cdef unsigned int num_descs = imgA_descs.shape[3]
    cdef unsigned int i
    cdef float dist = 0
       
    for i in range(num_descs):       
        dist += (imgA_descs[yA,xA,0,i] - imgB_descs[yB,xB,0,i])*(imgA_descs[yA,xA,0,i] - imgB_descs[yB,xB,0,i])
        if dist >= dbest:
            break
        
    return dist

@cython.boundscheck(False)
cdef float dist_nobest(numpy.ndarray[DTYPE_F_t,ndim=4] imgA_descs,numpy.ndarray[DTYPE_F_t,ndim=4] imgB_descs,int yA,int xA,int yB,int xB):
 
    cdef unsigned int num_descs = imgA_descs.shape[3]
    cdef unsigned int i
    cdef float dist = 0
       
    for i in range(num_descs):       
        dist += (imgA_descs[yA,xA,0,i] - imgB_descs[yB,xB,0,i])*(imgA_descs[yA,xA,0,i] - imgB_descs[yB,xB,0,i])
        
    return dist

@cython.boundscheck(False)
cdef numpy.ndarray[DTYPE_F_t,ndim=2] batch_dist(numpy.ndarray[DTYPE_F_t,ndim=4] imgA_descs,numpy.ndarray[DTYPE_F_t,ndim=4] imgB_descs,numpy.ndarray[DTYPE_t,ndim=3] ann):
    
    cdef numpy.ndarray[DTYPE_F_t,ndim=2] annd 
    cdef unsigned int x,y
    
    annd = numpy.zeros((imgA_descs.shape[0],imgA_descs.shape[1]),dtype=numpy.float32)
    for y in xrange(imgA_descs.shape[0]):
        for x in xrange(imgA_descs.shape[1]):
            annd[y,x] = dist_nobest(imgA_descs,imgB_descs,y,x,ann[y,x,1],ann[y,x,0])

    return annd   

@cython.boundscheck(False)    
# numpy.ndarray[DTYPE_t,ndim=3] 
cdef patchmatch_loop(numpy.ndarray[DTYPE_F_t,ndim=4]imgA_descs,numpy.ndarray[DTYPE_F_t,ndim=4] imgB_descs,numpy.ndarray[DTYPE_t,ndim=3] ann,unsigned int pm_iters, unsigned int rs_start):

    cdef int h,w
    cdef numpy.ndarray[DTYPE_F_t,ndim=2] annd
    cdef int pm_iter
    cdef int xstart, ystart, xend, yend, xchange, ychange
    cdef unsigned int x,y
    cdef unsigned int xbest, ybest
    cdef unsigned int neig_x, neig_y, new_x, new_y
    cdef float mag,dbest,dnew
    cdef unsigned int xmin,xmax,ymin,ymax
    cdef unsigned int xp,yp

    h = imgA_descs.shape[0]
    w = imgA_descs.shape[1]    

#    print "Creating annd"
    annd = batch_dist(imgA_descs,imgB_descs,ann)

    for pm_iter in range(pm_iters):
#        print "pm_iter",pm_iter,"/",pm_iters
        ystart = 0; yend = h; ychange = 1
        xstart = 0; xend = w; xchange = 1
        if pm_iter % 2 == 1: # odd
            ystart = h-1 ; yend = -1 ; ychange = -1
            xstart = w-1 ; xend = -1 ; xchange = -1
        
        for y in range(ystart,yend,ychange):
            for x in range(xstart,xend,xchange):
               
                # current (best) guess
                xbest,ybest = ann[y,x,:]
                dbest = annd[y,x]
                
                # propogation
                neig_x = x-xchange
                neig_y = y                
                
                if neig_x >=0 and neig_x <w:
                    new_x = ann[neig_y,neig_x,0] + xchange
                    new_y = ann[neig_y,neig_x,1] 
                
                    if (new_x >= 0) and (new_x < w):
                        dnew = dist(imgA_descs,imgB_descs,y,x,new_y,new_x,dbest)
                        if dnew < dbest:
                            xbest = new_x
                            ybest = new_y
                            dbest = dnew
                                
                neig_x = x
                neig_y = neig_y - ychange
                                
                if neig_y >=0 and neig_y <h:
                    new_x = ann[neig_y,neig_x,0]
                    new_y = ann[neig_y,neig_x,1] + ychange
                    if new_y >=0 and new_y<h:
                        dnew = dist(imgA_descs,imgB_descs,y,x,new_y,new_x,dbest)
                        if dnew < dbest:
                            xbest = new_x
                            ybest = new_y
                            dbest = dnew
                
           
                # random search
                mag = rs_start
                while mag>=1:
                    xmin = int_max(<int>(xbest-mag),0)
                    ymin = int_max(<int>(ybest-mag),0)
                    xmax = int_min(<int>(xbest+mag+1),w-1)
                    ymax = int_min(<int>(ybest+mag+1),h-1)
                    xp = xmin + numpy.random.randint(0,xmax-xmin)
                    yp = ymin + numpy.random.randint(0,ymax-ymin)
                    dnew = dist(imgA_descs,imgB_descs,y,x,yp,xp,dbest)
                    if dnew < dbest:
                        xbest = xp
                        ybest = yp
                        dbest = dnew
                    mag /= 2
                    
                ann[y,x,0] = xbest
                ann[y,x,1] = ybest
                annd[y,x] = dbest
    
    return ann,annd
    

@cython.boundscheck(False)    
def patchmatch(numpy.ndarray[DTYPE_F_t,ndim=4]imgA_descs,numpy.ndarray[DTYPE_F_t,ndim=4] imgB_descs,numpy.ndarray[DTYPE_t,ndim=3] random_ann,unsigned int pm_iters, unsigned int rs_start):

    cdef numpy.ndarray[DTYPE_t,ndim=3] ann
    
#    print "Starting patchmatch_loop"
    ann,annd = patchmatch_loop(imgA_descs,imgB_descs,random_ann,pm_iters,rs_start)
    
    return ann,annd
       

        



