from PIL import Image
import numpy as np
import glob
from tqdm import tqdm
import time
import json
import os
from argparse import ArgumentParser
import preprocessing_situ_all_data

def main(file_loc, storage_loc, prefix, _samples_per_record, _K, _T, _image_size, _seq_length, _step_size):
    #setup preprocessing script.
    preprocessing_situ_all_data.set_dest_path(storage_loc, _samples_per_record, _K, _T, _image_size, _seq_length, _step_size)
    preprocessing_situ_all_data.update_episode_reset_globals(prefix)
    ind = 0
    for file in tqdm(sorted(glob.glob(file_loc+'gridmap_*_occupancy.png'))):
        ind += 1

        #extract gridmap file number
        tmp = file.split('/')[-1]
        tmp = tmp.split('.')[0]
        nr = tmp.split('_')[1]

        #create location variables
        gridmap = file_loc+"gridmap_"+str(nr)+"_occupancy.png"
        segmentation = file_loc+"gridmap_"+str(nr)+"_stereo_cnn.png"
        rgb = file_loc+"gridmap_"+str(nr)+"_stereo_img.png"

        #blank depth since not provided by MBRDNA
        dep = np.zeros((1920,640))

        #write odometry.txt by extracting values from meta_data.json
        with open(file_loc+"gridmap_"+str(nr)+"_meta_data.json", 'r') as fp:
            try:
                obj = json.load(fp)
                speed = obj['dynamics']['speed']['value']
                yaw_rate = obj['dynamics']['yawrate']['value']
            except:
                continue

        #pass frames to preprocessing script that buffers them
        #and bundles to TFRecord once enough frames have been received.
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
    parser.add_argument("--samples-per-record", type=int, dest="_samples_per_record", default=20,
                        help="Number of sequences per TFRecord.")
    parser.add_argument("--K", type=int, dest="_K", default=9,
                        help="Number of frames to observe before prediction.")
    parser.add_argument("--T", type=int, dest="_T", default=10,
                        help="Number of frames to predict.")
    parser.add_argument("--image-size", type=int, dest="_image_size", default=96,
                        help="Size of grid map.")
    parser.add_argument("--seq-length", type=int, dest="_seq_length", default=40,
                        help="How many frames per Sequence. Has to be at least K+T+1.")
    parser.add_argument("--step-size", type=int, dest="_step_size", default=5,
                        help="Number of frames to skip between sequences.")

    args = parser.parse_args()
    main(**vars(args))
