import numpy
import argparse
import patchbatch
import glob
import kittitool
import pb_utils as utils

def bench_kitti(images_path, GT_path, model_name, patch_size, batch_size):
    """ Used for easily benchmarking kitti, using kitti's file structure
        images_path - images path, with image pairs looking like: 000000_10.png, 000000_11.png for example
        GT_path - ground truth path, looking like 000000_10.png
        model_name - PatchBatch model to use

        Retruns nothing, prints results to screen """

    images_list = sorted(glob.glob(images_path + '/*.png'))
    gt_list = sorted(glob.glob(GT_path + '/*.png'))

    for img1_filename, img2_filename, gt_filename in zip(images_list[::2], images_list[1::2], gt_list):
        print 'Analyzing', img1_filename.split('/')[-1],img2_filename.split('/')[-1]
        gt_flow = kittitool.flow_read(gt_filename)
        pb_flow = patchbatch.calc_flow(img1_filename, img2_filename, model_name, None, patch_size, batch_size, False)
        utils.benchmark_flow(pb_flow, gt_flow, debug=False)


def main(patch_size=51, batch_size=256):
    parser = argparse.ArgumentParser(description = 'PatchBatch KITTI benchmark pipeline')
    parser.add_argument('images_path', help = 'input images path')
    parser.add_argument('gt_path', help = 'ground truth path')
    parser.add_argument('model_name', help = 'PatchBatch Model Name')

    parser = parser.parse_args()

    if 'SPCI' in parser.model_name:
        patch_size = 71
        batch_size = 255

    bench_kitti(parser.images_path, parser.gt_path, parser.model_name, patch_size, batch_size)


if __name__ == '__main__':
    main()
