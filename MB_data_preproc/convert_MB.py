from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import time
import json
import os
from argparse import ArgumentParser
import preprocessing_situ_all_data
#file_loc = '/mnt/ds3lab-scratch/lucala/new_dataset/all_data/'
#storage_loc = '/mnt/ds3lab-scratch/lucala/process_MB_large_data_format/tfrecords/'
def main(file_loc, storage_loc, prefix):
    preprocessing_situ_all_data.set_dest_path(storage_loc)
    preprocessing_situ_all_data.update_episode(prefix)
    ind = 0
    for file in tqdm(sorted(glob.glob(file_loc+'gridmap_*_occupancy.png'))):
        ind += 1
        #if ind < 23790: #so as not to start from beginning again in case of failure
        #    continue
        tmp = file.split('/')[-1]
        tmp = tmp.split('.')[0]
        nr = tmp.split('_')[1]
        #print(nr)
        gridmap = file_loc+"gridmap_"+str(nr)+"_occupancy.png"
        segmentation = file_loc+"gridmap_"+str(nr)+"_stereo_cnn.png"
        rgb = file_loc+"gridmap_"+str(nr)+"_stereo_img.png"
        dep = np.zeros((1920,640))
        with open(file_loc+"gridmap_"+str(nr)+"_meta_data.json", 'r') as fp:
            try:
                obj = json.load(fp)
                speed = obj['dynamics']['speed']['value']
                yaw_rate = obj['dynamics']['yawrate']['value']
            except:
                continue
        #img, rgb, depth, segmentation, yaw_rate, speed
        #print(speed)
        #print(yaw_rate)
        preprocessing_situ_all_data.main(Image.open(gridmap),
            np.asarray(Image.open(rgb).resize((1920,640),Image.ANTIALIAS)), dep,
            np.asarray(Image.open(segmentation).resize((1920,640),Image.ANTIALIAS)),
            yaw_rate, speed)

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--file-loc", type=str, dest="file_loc", default="./data/",
                        help="where is the data located?")
    parser.add_argument("--storage-loc", type=str, dest="storage_loc", default="./tfrecords/",
                        help="where should tfRecords be stored?")
    parser.add_argument("--prefix", type=str, dest="prefix", default="data",
                        help="string prepended to tfRecord name.")

    args = parser.parse_args()
    main(**vars(args))
