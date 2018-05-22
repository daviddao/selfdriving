import numpy as np
import scipy.misc
from scipy.ndimage import imread
from glob import glob 
import numpy
import os
from argparse import ArgumentParser
import time
from tqdm import tqdm

def main(in_path, out_path, thresh, occlusion_path = None, overwrite=False):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    occ_imgs = sorted(glob(os.path.join(in_path,"*")))

    counter = 0
    start_time = time.time()
    for img_path in tqdm(occ_imgs):
        counter += 1
        shouldSaveImage = True
        if (time.time() - start_time) > 1:
            #print("Processed "+str(counter)+" images, next = "+img_path)
            start_time = time.time()
        img_name = img_path.rsplit('/',2)[-1]
        save_path = os.path.join(out_path, img_name)
        if overwrite or not os.path.isfile(save_path):
            img = np.array(imread(img_path)).astype(np.float32)
            if len(img.shape) == 3:
                img = np.mean(img, axis=2)
            img = (img < thresh) * 1.0

            img = draw_ego_vehicle(img)
            
            if occlusion_path is not None:
                occl_file = os.path.join(occlusion_path, img_name.split(".",-1)[0] + "_occlusion.pgm")
                try:
                    occl = imread(occl_file)
                    img = np.multiply(img, occl)
                except IOError:
                    shouldSaveImage = False
            if shouldSaveImage:
                scipy.misc.toimage(img).save(save_path)
            else:
                print("File not saved: "+save_path)

def draw_ego_vehicle(img):
    image_size = img.shape[0]
    VEHICLE_WIDTH = image_size * 4 // 96
    VEHICLE_HEIGHT = image_size * 7 // 96
    x_start = image_size // 2 - VEHICLE_WIDTH // 2
    x_end = x_start + VEHICLE_WIDTH
    y_start = image_size // 2
    y_end = y_start + VEHICLE_HEIGHT
    img[y_start:y_end, x_start:x_end] = 1
    return img

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
                            required=True, help="Path to directory of occupancy images")
    parser.add_argument("--outpath", type=str, dest="out_path",
                            required=True, help="Path to directory where the masks should be saved")
    parser.add_argument("--thresh", type=int, dest="thresh",
                            default=96, help="Threshold for mean of pixels over channels (between 0 and 255)")
    parser.add_argument("--occlpath", type=str, dest="occlusion_path",
                            default=None, help="Path to occlusion maps which should be used for blending out. None means no masking")
    parser.add_argument("--overwrite", type=str2bool, dest="overwrite",
                            default=False, help="Determines whether existing files in --outpath should be overwritten or skipped")


    args = parser.parse_args()
    main(**vars(args))