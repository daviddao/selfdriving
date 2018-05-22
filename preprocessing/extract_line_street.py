import os
import numpy as np 
from glob import glob 
from scipy.ndimage import imread
import scipy.misc
from argparse import ArgumentParser
from tqdm import tqdm

def main(in_path, out_path, overwrite=False):
    horizon_maps = sorted(glob(os.path.join(in_path,"*_horizon_map.*")))
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    for img_path in tqdm(horizon_maps):
        line_save_path = os.path.join(out_path, img_path.split("/",-1)[-1].split(".",-1)[-1] + "_lines.png")
        road_save_path = os.path.join(out_path, img_path.split("/",-1)[-1].split(".",-1)[-1] + "_road.png")
        if overwrite or (not os.path.isfile(line_save_path) and not os.path.isfile(road_save_path)):
            img = imread(img_path)
            lines = (img[:,:,0] > 128) * 255
            road = (img[:,:,2] > 128) * 255
            scipy.misc.toimage(lines, cmin=0, cmax=255).save(line_save_path)
            scipy.misc.toimage(road, cmin=0, cmax=255).save(road_save_path)


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
    parser.add_argument("--overwrite", type=str2bool, dest="overwrite",
                        default=False, help="Path to directory where images should be saved")

    args = parser.parse_args()
    main(**vars(args))
