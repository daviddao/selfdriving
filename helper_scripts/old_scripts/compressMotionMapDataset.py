"""
Script for compressing frame sequences for input to MCnet or GAN
"""
import numpy as np
from glob import glob
from scipy.ndimage import imread
from PIL import Image
import os

velocity_list = sorted(glob("/lhome/phlippe/dataset/TwoHourSequence_crop/Sequence2/velmask_128x128/*"))
occlusion_path = "/lhome/phlippe/dataset/TwoHourSequence_crop/Sequence2/occlusion_seq2_128x128/"
MOTION_ORIENT_PATH = '/lhome/phlippe/dataset/Encoding2/velocity_orientation_128x128.png'
MOTION_INTENS_PATH = '/lhome/phlippe/dataset/Encoding2/one_channel_map_128x128.png'
"""
vel_list = sorted(
    glob("/lhome/phlippe/dataset/TwoHourSequence_crop/cropped_velocity/*.ppm"))
vel_base_path = "/lhome/phlippe/dataset/TwoHourSequence_crop/cropped_velocity/gridmap_"
"""
save_dir = "/lhome/phlippe/dataset/TwoHourSequence_crop/velHorizonScaled96x96/"
seq_length = 81
image_size = 128
crop_size = 96
step_size = 1


def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames /= (255 / 2)
    new_frames -= 1

    return new_frames

def crop_frames(frames):
    height = frames.shape[0]
    width = frames.shape[1]
    start_x = (width - crop_size)/2
    start_y = 0
    end_x = start_x + crop_size
    end_y = start_y + crop_size

    cropped_frames = frames[start_y:end_y, start_x:end_x]
    return cropped_frames

def create_default_element():
    element = np.zeros([crop_size, crop_size, seq_length*2+4],dtype=np.float32)
    motion_orientation = imread(MOTION_ORIENT_PATH)
    motion_orientation = normalize_frames(crop_frames(motion_orientation))
    motion_intens = imread(MOTION_INTENS_PATH)
    motion_intens = normalize_frames(crop_frames(motion_intens))
    motion_intens = np.reshape(motion_intens, [motion_intens.shape[0], motion_intens.shape[1],1])
    element[:,:,0:3] = motion_orientation
    element[:,:,3:4] = motion_intens
    return element

def get_frame_number(frame_path):
    return frame_path.split("/",-1)[-1].split(".",-1)[0].split("_")[1]

def get_occlusion_path(frame_path):
    return os.path.join(occlusion_path, "gridmap_" + get_frame_number(frame_path) + "_occupancy_occlusion.pgm")

def read_frame(image_path):
    velocity_map = imread(image_path)
    occlusion_map = imread(get_occlusion_path(image_path))
    return np.stack([velocity_map, occlusion_map], axis=2)

all_clips = [create_default_element() for i in xrange(seq_length)]
for file_index in xrange(len(velocity_list) - seq_length):
    frame = read_frame(velocity_list[file_index])
    print velocity_list[file_index]

    norm_frame = normalize_frames(crop_frames(frame))
    for clip_index in xrange(len(all_clips)):
        all_clips[clip_index][:, :, 4 + clip_index*2 : 4 + (clip_index+1)*2 ] = norm_frame
        #all_clips[clip_index][:, :, clip_index, 3:] = norm_vel
    if file_index >= seq_length - 1 and file_index % step_size == 0:
        np.savez_compressed(save_dir + "seq_" +
                            str(file_index).zfill(7)+"_seq2", all_clips[-1])
        print "Save " + save_dir + "seq_" + str(file_index).zfill(7)
    del all_clips[-1]
    all_clips.insert(0, create_default_element())
