PatchBatch - a Batch Augmented Loss for Optical Flow
====================================================

This is an initial commit implementing *PatchBatch - a Batch Augmented Loss for Optical Flow* [link](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gadot_PatchBatch_A_Batch_CVPR_2016_paper.html)
The code was developed on Ubuntu 14.04, using Theano+Lasagne+OpenCV. You can see the performance it achieved on the [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow), [KITTI2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [MPI-Sintel](http://sintel.is.tue.mpg.de/) optical flow scoreboards.

For now only the ACCURATE network weights have been uploaded and implemented in this repository, the FAST weights will follow.

Installation Instructions
-------------------------
1. Download and compile OpenCV 2.4.10, with python2.7 support
2. Create a python (2.7) virtualenv, by typing: `virtualenv --no-site-packages env`
3. Copy the cv2.so file which was generated in step 1 into `env/lib/python2.7/site-packages`
4. Clone this repository by typing: `git clone https://github.com/DediGadot/PatchBatch`
5. Install all the python packages described in Requirements.txt by typing: `pip install -r Requirements.txt`
6. Make sure to configurae Theano per your needs (GPU usage preferred)

Usage
-----
To run the PatchBatch pipeline, use the following syntax:
`python patchbatch.py <img1_filename> <img2_filename> <output_path> [optional -bidi]`
If the output_path does not exist, it will be created and in it the following files:
flow.pickle - a <h,w,3> numpy array with channel 0,1,2 equals U, V, valid(i.e. 0 is not valid, 1 is valid) components of the flow field.
If the `--bidi` flag is invoked, the code will compute 2 flow fields: img1->img2 and img2->img1 and will mark as 'invalid' all correspondences with inconsistent matchings (i.e. >1 pixels apart).
