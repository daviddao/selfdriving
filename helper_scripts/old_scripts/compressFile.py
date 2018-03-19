"""
Script for compressing frame sequences for input to MCnet or GAN
"""
import numpy as np
from glob import glob
from scipy.ndimage import imread
from PIL import Image

file_list = sorted(glob(
    "/lhome/phlippe/dataset/TwoHourSequence_crop/Test256x256/*.png"))
"""
vel_list = sorted(
    glob("/lhome/phlippe/dataset/TwoHourSequence_crop/cropped_velocity/*.ppm"))
vel_base_path = "/lhome/phlippe/dataset/TwoHourSequence_crop/cropped_velocity/gridmap_"
"""
save_dir = "/lhome/phlippe/dataset/TwoHourSequence_crop/compressedTest256x256/"
seq_length = 81
image_size = 128
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

all_clips = [np.zeros([image_size, image_size, seq_length, 3],
                      dtype=np.float32) for i in xrange(seq_length)]
for file_index in xrange(len(file_list) - seq_length):
    frame = imread(file_list[file_index])
    print file_list[file_index]
    #vel_file = vel_base_path + file_list[file_index].split("/", -1)[-1].split(".", -1)[0].split("_", -1)[1]+"_velocity.ppm"
    #print vel_file
    #vel = np.array(Image.open(vel_file))
    norm_frame = normalize_frames(frame[:, :, :3])
    #norm_vel = normalize_frames(vel)
    for clip_index in xrange(len(all_clips)):
        all_clips[clip_index][:, :, clip_index, :3] = norm_frame
        #all_clips[clip_index][:, :, clip_index, 3:] = norm_vel
    if file_index >= seq_length - 1 and file_index % step_size == 0:
        np.savez_compressed(save_dir + "seq_" +
                            str(file_index).zfill(7), all_clips[-1])
        print "Save " + save_dir + "seq_" + str(file_index).zfill(7)
    del all_clips[-1]
    all_clips.insert(0, np.zeros(
        [image_size, image_size, seq_length, 3], dtype=np.float32))
