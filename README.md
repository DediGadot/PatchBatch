PatchBatch - a Batch Augmented Loss for Optical Flow
====================================================
This is an initial commit implementing **PatchBatch - a Batch Augmented Loss for Optical Flow** by Dedi Gadot and Lior Wolf from Tel Aviv University [(link)](http://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Gadot_PatchBatch_A_Batch_CVPR_2016_paper.html), published at **CVPR, 2016**.  

PatchBatch achieved state-of-the-art results in 2016 on the KITTI (2012+2015) Optical Flow benchmarks and was ranked 6th on
MPI-Sintel, though ranked 1st for small displacements.

The code was developed on Ubuntu 14.04, using Theano+Lasagne+OpenCV. You can see the performance it achieved on the [KITTI2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=flow), [KITTI2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=flow) and [MPI-Sintel](http://sintel.is.tue.mpg.de/) optical flow scoreboards.  

For now only the ACCURATE networks have been uploaded, the FAST network will follow.

**UPDATE** - New trained models (SPCI) are available. Based on **Optical Flow Requires Multiple Strategies (but only one network)** by Tal Schuster, Lior Wolf and Dedi Gadot [(link)](https://arxiv.org/abs/1611.05607v1).

Installation Instructions
-------------------------
1. Download and compile OpenCV 2.4.10, with python2.7 support
2. Create a python (2.7) virtualenv, by typing: `virtualenv --no-site-packages env`
3. Copy the cv2.so file which was generated in step 1 into `env/lib/python2.7/site-packages`
4. Clone this repository by typing: `git clone https://github.com/DediGadot/PatchBatch`
5. Install all the python packages described in Requirements.txt by typing: `pip install -r Requirements.txt`
6. Make sure to configure Theano to your needs (GPU usage preferred)

The PatchBatch Pipeline
-----------------------
The PatchBatch pipeline consists of the following steps:

1. **Input**: two grayscale images, with the same shape  
2. Calculate descriptors (per each pixel in both images) using the PatchBatch CNN, i.e calculate a [h,w,#dim] tensor per
   image  
3. Find correspondences between both descriptor tensors using PatchMatch, with an L2 cost function  
4. Eliminate incorrect assignments using a bidirectional consistency check  
5. **(Not yet implemented in this repository)** Use the L2 costs + EpicFlow algorithm to interpolate the sparse optical
   flow field into a dense one (we used the default parameters of EpicFlow)
6. **Outputs**: A->B optical flow field, (optional) descriptors file, cost assignment file

Usage
-----
To run the PatchBatch pipeline, use the following syntax:  
`python patchbatch.py <img1_filename> <img2_filename> <model_name> <output_path> [optional -bidi] [optional --descs]`  

Currently supported models:
* KITTI2012_CENTSD_ACCURATE
* KITTI2015_CENTSD_ACCURATE
* KITTI2012_SPCI
* KITTI2015_SPCI

If the output_path does not exist, it will be created. In it will be placed the following:  
* flow.pickle - 
  * A [h,w,3] numpy array with channel 0,1,2 being U, V, valid flag components of the flow field 
  * If the `-bidi` flag is invoked, the code will compute 2 flow fields: img1->img2 and img2->img1 and will mark as 'invalid' all correspondences with inconsistent matchings (i.e. >1 pixels apart)
* cost.pickle - 
  * A [h,w] numpy array containing the matching cost per match
* (If the --descs option was used) descs.pickle - 
  * A list with two [h,w,#d] numpy arrays, the first contains descriptors per each pixel of img1, and the second the same for img2

You can also use `benchmark_kitti.py` to run a full benchmark on a folder with KITTI images.

For now, the EpicFlow extension is not yet implemented - so what you're getting is a pure PatchBatch descriptors + PatchMatch result.

Credits
-------
The PatchBatch pipeline couldn't be achieved without the following great software pieces:
* [Theano](https://github.com/Theano/Theano)  
* [Lasagne](https://github.com/Lasagne/Lasagne)  

We also used the following toolkit for visualization:
* [OpticalFlowToolkit](https://github.com/liruoteng/OpticalFlowToolkit)

MIT License
-------
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
