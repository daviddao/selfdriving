
import os
import numpy as np
from glob import glob
from scipy.ndimage import imread
import scipy.misc

IMAGE_DIR = "/lhome/phlippe/dataset/TwoHourSequence_crop/Sequence1/img128x128/"
OCCLUSION_DIR = "/lhome/phlippe/dataset/TwoHourSequence_crop/Sequence1/occlusion_128x128/"
SAVE_DIR = "/lhome/phlippe/dataset/TwoHourSequence_crop/Sequence1/occluded_img_128x128/"

image_files = sorted(glob(os.path.join(IMAGE_DIR, '*')))
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

for image_path in image_files:
    image_number = image_path.split("/",-1)[-1].split(".",-1)[0].split("_",-1)[1]
    occlusion_path = os.path.join(OCCLUSION_DIR, "gridmap_"+image_number+"_occupancy_occlusion.pgm")
    save_path = os.path.join(SAVE_DIR, "gridmap_"+image_number+"_occluded.png")

    if os.path.isfile(occlusion_path) and not os.path.isfile(save_path):
        print image_path
        img = imread(image_path)[:,:,:3]
        occlusion_map = imread(occlusion_path)/255.0
        occlusion_map = np.stack([occlusion_map]*3, axis=2)
        
        occluded_img = np.multiply(img, occlusion_map)
        scipy.misc.toimage(occluded_img).save(save_path)


