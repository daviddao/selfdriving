"""
Script for cropping and resizing gridmap images
"""
from PIL import Image
import PIL
from resizeimage import resizeimage as resImg 
from glob import glob
import os

DATA_SETS = sorted(glob("/lhome/phlippe/data/ml_data/trajectory_learning/20170517_around_maude_CNNDataWriter_dump_20170518/"))  # Directory of folders in which the images should be processed
SAVE_PATH = "/lhome/phlippe/dataset/Maude/img96x96/"  # Directory where to save the processed images
CROP_SIZE = 456     # Size to which the map should be cropped
SCALE_SIZE = 96    # Size to which the cropped map should be resized

for folder in DATA_SETS:
    print folder
    new_folder = SAVE_PATH# + folder.split("/",-1)[-1]
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    
    #new_folder = SAVE_PATH
    images = sorted(glob(os.path.join(folder,"*.p?m")))
    counter = 0
    for image_path in images:
        counter += 1
        img_save_path = os.path.join(new_folder,image_path.split("/",-1)[-1])
        # if os.path.isfile(img_save_path):
        #    continue
        print str(counter)+"|"+str(len(images))+": "+image_path
        with open(image_path, "r+b") as f:
            with Image.open(f) as img:
                #img = resImg.resize_width(img, 1024)
                crop_img = resImg.resize_crop(img, [CROP_SIZE, CROP_SIZE])
                scaled_img = resImg.resize_width(crop_img, SCALE_SIZE)
                scaled_img.save(img_save_path) 
                #img = img.resize((SCALE_SIZE, SCALE_SIZE), PIL.Image.ANTIALIAS)
                #img.save(new_folder + "/" + image_path.split("/",-1)[-1])
