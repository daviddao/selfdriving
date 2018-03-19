import numpy as np
import scipy.misc
from scipy.ndimage import imread
from glob import glob 
import numpy
import os
from argparse import ArgumentParser
import time

def main(in_path, out_path, thresh, occlusion_path = None):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    occ_imgs = sorted(glob(os.path.join(in_path,"*")))

    counter = 0
    start_time = time.time()
    for img_path in occ_imgs:
        counter += 1
        shouldSaveImage = True
        if (time.time() - start_time) > 1:
            print "Processed "+str(counter)+" images, next = "+img_path
            start_time = time.time()
        img_name = img_path.rsplit('/',2)[-1]
        save_path = os.path.join(out_path, img_name)

        img = np.array(imread(img_path)).astype(np.float32)
        if len(img.shape) == 3:
            img = np.mean(img, axis=2)
        img = (img < thresh) * 1.0
        
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
            print "File not saved: "+save_path


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

    args = parser.parse_args()
    main(**vars(args))