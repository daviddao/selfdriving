import os
import numpy as np
from glob import glob
from scipy.ndimage import imread
import scipy.misc
from argparse import ArgumentParser
from tqdm import tqdm

def main(mask1_path, mask2_path, out_path, mask1_suffix, mask2_suffix, out_suffix, overwrite=False):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    mask1_list = sorted(glob(os.path.join(mask1_path, "*"+mask1_suffix)))
    for mask1_file in tqdm(mask1_list):
        file_prefix = mask1_file.split("/",-1)[-1][:-len(mask1_suffix)]
        mask2_file = os.path.join(mask2_path, file_prefix+mask2_suffix)
        save_path = os.path.join(out_path, file_prefix+out_suffix)
        if overwrite or not os.path.isfile(save_path):
            mask1 = np.array(imread(mask1_file))
            mask2 = np.array(imread(mask2_file))
            
            if np.max(mask1) > 0:
                mask1 = mask1 * 1.0 / np.max(mask1)
            else:
                print("Mask 1 has no unoccluded space")
            if np.max(mask2) > 0:
                mask2 = mask2 * 1.0 / np.max(mask2)
            else:
                print("Mask 2 \""+mask2_file+"\" has no unoccluded space")
            
            final_mask = np.multiply(mask1, mask2)
            scipy.misc.toimage(final_mask, cmin=0, cmax=1).save(save_path) 

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mask1", type=str, dest="mask1_path",
                        required=True, help="Path to directory of the images")
    parser.add_argument("--mask2", type=str, dest="mask2_path",
                        required=True, help="Path to occlusion maps which should be used for masking out")
    parser.add_argument("--outpath", type=str, dest="out_path",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--mask1_suffix", type=str, dest="mask1_suffix",
                        required=True, help="Path to occlusion maps which should be used for masking out")
    parser.add_argument("--mask2_suffix", type=str, dest="mask2_suffix",
                        required=True, help="Path to occlusion maps which should be used for masking out")
    parser.add_argument("--out_suffix", type=str, dest="out_suffix",
                        required=True, help="Path to occlusion maps which should be used for masking out")
    parser.add_argument("--overwrite", type=str2bool, dest="overwrite",
                        default=False, help="Determines whether existing files in --outpath should be overwritten or skipped")
    


    args = parser.parse_args()
    main(**vars(args))