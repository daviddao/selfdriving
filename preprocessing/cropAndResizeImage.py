"""
Script for cropping and resizing gridmap images
"""
from PIL import Image, ImageOps
import PIL
#from resizeimage import resizeimage as resImg 
from argparse import ArgumentParser
import glob
import scipy.misc
import numpy as np 
import os
from tqdm import tqdm

def main(frame_files, save_path, crop_size, scale_size, prescale=-1, overwrite=False, crop_start_x = -1, crop_start_y = -1, suffix=None):
	print(frame_files)
	new_folder = save_path
	if not os.path.exists(new_folder):
		os.makedirs(new_folder)

	for img_path in tqdm(glob.glob(frame_files)):
		img_save_path = os.path.join(new_folder,img_path.split("/",-1)[-1])
		
		if suffix is not None:
			img_save_path = img_save_path.split(".",-1)[-1] + suffix
		
		#with open(img_path, "r", encoding="latin1") as f:
		#    print(f)
		with Image.open(img_path) as img:
			if prescale > 0:
				#img = resImg.resize_width(img, prescale)
				#img = img.resize((prescale,hsize), Image.ANTIALIAS)
				img.thumbnail((prescale, prescale), PIL.Image.ANTIALIAS)
			# crop_img = img.crop((START_X, START_Y,START_X+CROP_SIZE, START_Y+CROP_SIZE))
			#crop_img = resImg.resize_crop(img, [crop_size, crop_size])
			crop_img = ImageOps.fit(img, [crop_size, crop_size])
			scaled_img = crop_img.resize((scale_size, scale_size), PIL.Image.ANTIALIAS)
			scaled_img.save(img_save_path) 
			#img = img.resize((SCALE_SIZE, SCALE_SIZE), PIL.Image.ANTIALIAS)
			#img.save(new_folder + "/" + image_path.split("/",-1)[-1])


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, dest="frame_files",
                        required=True, help="Path to directory of the original images")
    parser.add_argument("--outpath", type=str, dest="save_path",
                        required=True, help="Path to directory where the processed images should be saved")
    parser.add_argument("--cropsize", type=int, dest="crop_size",
                        required=True, help="Size to which the images should be cropped. See --cropstartx and --cropstarty for more details.")
    parser.add_argument("--scalesize", type=int, dest="scale_size",
                        required=True, help="Size to which the cropped image should be scaled.")
    parser.add_argument("--cropstartx", type=int, dest="crop_start_x",
                        default=-1, help="Start pixel on x axis for cropping. Default crops around the center")
    parser.add_argument("--cropstarty", type=int, dest="crop_start_y",
                        default=-1, help="Start pixel on y axis for cropping. Default crops around the center")
    parser.add_argument("--overwrite", type=str2bool, dest="overwrite",
                        default=False, help="Determines whether existing files in --outpath should be overwritten or skipped")
    parser.add_argument("--prescale", type=int, dest="prescale",
                        default=-1, help="Size to which the original image should be scaled before cropping. -1 means no scaling.")
    parser.add_argument("--suffix", type=str, dest="suffix",
                        default=None, help="Suffix which the saved files should have. Default: the same as the original one")
    
    args = parser.parse_args()
    main(**vars(args))