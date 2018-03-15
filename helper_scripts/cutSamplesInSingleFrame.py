"""
Script for cutting the single image with multiple frames into single frames and generating GIFs of them (used for MCNet training results)
"""
import numpy as np 
from glob import glob
from scipy.ndimage import imread
import scipy.misc
import os
import imageio

# Images which should be processed. The new files will be saved in a directory with the same file name
SAMPLE_FILE_PATHS = sorted(glob("/lhome/phlippe/Documents/Python/iclr2017mcnet/samples/GRIDMAP_MCNET_image_size=64_K=10_T=10_batch_size=32_alpha=1.0_beta=0.02_lr=1e-05/*.png"))


def cutSampleInSingleFrames(sample, row_count, column_count, image_base_path):
    row_step = sample.shape[0] / row_count
    column_step = sample.shape[1] / column_count
    step_file = image_base_path.rsplit("/",-1)[-2].split("_",-1)[-1]

    for row in xrange(row_count):
        all_frames_in_row = []
        if row is 0:
            prefile = "_gen_"+step_file+"_"
        else:
            prefile = "_gt_"+step_file+"_"
        for column in xrange(column_count):
            frame = sample[row * row_step: (row + 1) * row_step, column * column_step : (column + 1) * column_step,:]
            file_path = image_base_path
            file_path += prefile
            file_path += str(column).zfill(2) + ".png"
            scipy.misc.toimage(frame).save(file_path)
            all_frames_in_row.append(frame)
        GIF_PATH = image_base_path + "GIF"+prefile+".gif"
        kargs = { 'duration': 0.25 }
        imageio.mimsave(GIF_PATH, all_frames_in_row, 'GIF', **kargs)

for sample_file in SAMPLE_FILE_PATHS:
    print sample_file
    sample = scipy.misc.imread(sample_file)
    new_folder = sample_file.rsplit("/",1)[0] + "/" + sample_file.rsplit("/",1)[-1].split(".",-1)[0] + "/"
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    cutSampleInSingleFrames(sample, 2, 10, new_folder + "frame")