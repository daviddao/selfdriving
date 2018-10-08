"""
Script for visualizing occlusion map on images. On RGB images the occlusion map is shown as black, for gray scale images
the background is gray, occupancy is white and occlusion is black.

Usage:
    python create_occlusion_images.py --inpath IN_PATH --outpath OUT_PATH --occlpath OCCLUSION_PATH [--horizonpath HORIZON_PATH]

Args:
    inpath - Path to directory where the images for are stored
    outpath - Path to directory in which the generated images should be saved
    occlpath - Path to directory where the occlusion maps belonging to the images are stored
    horizonpath - Path to directory where the horizon maps are stored. Optional, is used as background for 1 channel images
"""
import os
import numpy as np
from glob import glob
from scipy.ndimage import imread
import scipy.misc
from argparse import ArgumentParser
from tqdm import tqdm


def main(in_path, out_path, occlusion_folder, horizon_folder="", overwrite=False):
    image_files = sorted(glob(os.path.join(in_path, '*')))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    counter = 0
    for image_path in tqdm(image_files):
        counter += 1
        image_number = image_path.split(
            "/", -1)[-1].split(".", -1)[0].split("_", -1)[1]
        occlusion_path = os.path.join(
            occlusion_folder, "gridmap_" + image_number + "_occupancy_occlusion.pgm")
        save_path = os.path.join(
            out_path, "gridmap_" + image_number + "_occluded.png")

        if os.path.isfile(occlusion_path) and (overwrite or not os.path.isfile(save_path)):
            img = imread(image_path)
            occlusion_map = imread(occlusion_path) / 255.0
            if len(img.shape) == 2:
                img = np.reshape(img, [img.shape[0], img.shape[1], 1])
            if img.shape[2] == 1:
                #img = np.concatenate([img, np.zeros([img.shape[0], img.shape[1], 1]), np.zeros([img.shape[0], img.shape[1], 1])], axis=2)
                img = np.concatenate([img]*3, axis=2)
                if horizon_folder is not None:
                    horizon_path = os.path.join(
                        horizon_folder, "gridmap_" + image_number + "_horizon_map.png")
                    if os.path.isfile(horizon_path):
                        horizon_map_img = imread(horizon_path)
                    else:
                        horizon_map_img = np.zeros(img.shape, dtype=np.float32)
                    horizon_map_img = horizon_map_img + (horizon_map_img < 10) * 255
                    img = img + (255 - img) / 255 * 0.5 * horizon_map_img
                else:
                    img = img + (255 - img) * 0.5

            occlusion_map = np.stack([occlusion_map] * 3, axis=2)
            occluded_img = np.multiply(img, occlusion_map)
            scipy.misc.toimage(occluded_img).save(save_path)
        else:
            if not os.path.isfile(occlusion_path):
                print("Could not find occlusion file " + occlusion_path)

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, dest="in_path",
                        required=True, help="Path to directory of the images")
    parser.add_argument("--outpath", type=str, dest="out_path",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--occlpath", type=str, dest="occlusion_folder",
                        required=True, help="Path to occlusion maps which should be used for masking out")
    parser.add_argument("--horizonpath", type=str, dest="horizon_folder",
                        default="", help="Path to horizon maps which should be used as background for 1 channel images")
    parser.add_argument("--overwrite", type=str2bool, dest="overwrite",
                        default=False, help="Determines if an existing image should be overwritten or skipped")

    args = parser.parse_args()
    main(**vars(args))
