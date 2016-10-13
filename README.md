PatchBatch - a Batch Augmented Loss for Optical Flow
====================================================

This is an initial commit implementing *PatchBatch - a Batch Augmented Loss for Optical Flow* [link](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gadot_PatchBatch_A_Batch_CVPR_2016_paper.html)
The code was developed on Ubuntu 14.04, using Theano+Lasagne+OpenCV. You can see the performance it achieved on the [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow), [KITTI2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [MPI-Sintel](http://sintel.is.tue.mpg.de/) optical flow scoreboards.

For now only the ACCURATE network weights have been uploaded and implemented in this repository, the FAST weights will follow.

Installation Instructions
-------------------------
1. Download and compile OpenCV 2.4.10, with python support
2. Create a python (2.7) virtualenv, by typing: `virtualenv --no-site-packages env`
3. Copy the cv2.so file which was generated in step 1 into `env/lib/python2.7/site-packages`
4. Clone this repository by typing: `git clone https://github.com/DediGadot/PatchBatch`
5. Install all the python packages described in Requirements.txt by typing: `pip install -r Requirements.txt`



