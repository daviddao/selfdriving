"""
Script for compressing frame sequences for input to MCnet or GAN
"""
import numpy as np
from glob import glob
from scipy.ndimage import imread
from PIL import Image
import os
from argparse import ArgumentParser
import time
from tqdm import tqdm


def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    new_frames //= (255 // 2)
    new_frames -= 1

    return new_frames


def create_default_element(image_size, seq_length, channel_size):
    element = np.zeros(
        [image_size, image_size, seq_length * channel_size], dtype=np.float32)
    return element


def get_frame_number(frame_path):
    return frame_path.split("/", -1)[-1].split(".", -1)[0].split("_",-1)[1]


def get_correlated_file(frame_path, directory_path, suffix):
    return os.path.join(directory_path, "gridmap_" + get_frame_number(frame_path) + suffix)


def read_frame(image_path, occl_path, occl_path_suffix, road_path, road_path_suffix, lines_path, lines_path_suffix, combined_mask_path, comb_path_suffix):
    occup_map = imread(image_path)
    occlusion_map = imread(get_correlated_file(image_path, occl_path, occl_path_suffix))
    #lines_map = imread(os.path.join(lines_path, get_correlated_file(image_path, lines_path, lines_path_suffix))) # "gridmap_1498528354613695_horizon_map_lines.ppm") ) #
    #road_map = imread(os.path.join(road_path, get_correlated_file(image_path, road_path, road_path_suffix))) # "gridmap_1498528354613695_horizon_map_road.ppm") )
    lines_map = imread(get_correlated_file(image_path, lines_path, lines_path_suffix)) # "gridmap_1498528354613695_horizon_map_lines.ppm") ) #
    road_map = imread(get_correlated_file(image_path, road_path, road_path_suffix)) # "gridmap_1498528354613695_horizon_map_road.ppm") )
    combined_mask_map = imread(
        get_correlated_file(image_path, combined_mask_path, comb_path_suffix))
    frame = np.stack([occup_map, occlusion_map, lines_map, road_map, combined_mask_map], axis=2)
    return frame #frame[52:52+96,36:36+96,:]


def read_matrix(image_path, matrix_path, file_suffix):
    if matrix_path is not None:
        matrix_file_path = get_correlated_file(image_path, matrix_path, file_suffix)
        matrix = np.load(matrix_file_path)['arr_0']
    else:
        matrix = np.zeros([3,2,3], dtype=np.float32)
        matrix[:,0,0] = 1
        matrix[:,1,1] = 1
    tf_matrix = np.zeros([3,8], dtype=np.float32)
    tf_matrix[:,0:3] = matrix[:,0,:]
    tf_matrix[:,3:6] = matrix[:,1,:]
    return tf_matrix


def get_occupancy_diff(clip):
    occup_clip = np.multiply((clip[:,:,::5] + 1)/2.0, (clip[:,:,4::5] + 1)/2.0)
    occup_diff = occup_clip[:,:,:-1] - occup_clip[:,:,1:]
    occup_diff = np.absolute(occup_diff)
    return np.sum(occup_diff) * 1.0 / occup_diff.shape[2]


def main(occup_path, occl_path, occl_path_suffix, road_path, road_path_suffix, lines_path, lines_path_suffix, 
	combined_mask_path, comb_path_suffix, transformation_path, transformation_suffix, out_path, seq_length, 
	step_size, image_size, transformation_only, out_suffix, split_number = 0, split_amount = 1):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    max_occup_diff = 0
    min_occup_diff = 100000
    mean_occup_diff = 0.0
    occup_list = sorted(glob(os.path.join(occup_path, "*")))
    all_clips = [create_default_element(
        image_size, seq_length, 5) for i in range(seq_length)]
    all_transformation = [np.zeros([seq_length, 3, 8], dtype=np.float32) for i in range(seq_length)]

    step_offset = int(round(1.0 * step_size / split_amount * split_number))

    start_time = time.time()
    for file_index in tqdm(range(len(occup_list) - seq_length)):
        if file_index == seq_length - 1:
            start_time = time.time()
        try:
            frame = read_frame(
                occup_list[file_index], occl_path, occl_path_suffix, road_path, road_path_suffix, lines_path, lines_path_suffix, combined_mask_path, comb_path_suffix)
            if transformation_only:
                frame = np.zeros([1,1,1])
            transform_matrix = read_matrix(occup_list[file_index], transformation_path, transformation_suffix)
        except IOError as e:
            print("Got IOError by reading file "+occup_list[file_index])
            print("I/O error({0}): {1}".format(e.errno, e.strerror))
            print(e)
            file_index -= 1
            occup_list.remove(occup_list[file_index])
            continue
        channel_size = frame.shape[2]

        norm_frame = normalize_frames(frame)
        for clip_index in range(len(all_clips)):
            if not transformation_only:
                all_clips[clip_index][:, :, clip_index *
                                      channel_size: (clip_index + 1) * channel_size] = norm_frame
            all_transformation[clip_index][clip_index,:,:] = transform_matrix
            #all_clips[clip_index][:, :, clip_index, 3:] = norm_vel
        if file_index >= seq_length - 1 and (file_index - step_offset) % step_size == 0:
            if not transformation_only:
                occup_diff = get_occupancy_diff(all_clips[-1])
                max_occup_diff = max(max_occup_diff, occup_diff)
                min_occup_diff = min(min_occup_diff, occup_diff)
                mean_occup_diff = mean_occup_diff + occup_diff
            if transformation_only or occup_diff > 6:
                comp_name = "seq_" + str(file_index).zfill(7)
                if out_suffix is not None:
                    comp_name = comp_name + "#" + out_suffix
                if not transformation_only:
                    np.savez_compressed(os.path.join(out_path, comp_name), all_clips[-1])
                np.savez_compressed(os.path.join(out_path, comp_name + "_transformation"), all_transformation[-1])
            else:
                print("Sequence not saved due to occupancy difference of " + str(occup_diff))

        del all_clips[-1]
        del all_transformation[-1]
        all_clips.insert(0, create_default_element(
            image_size, seq_length, channel_size))
        all_transformation.insert(0, np.zeros([seq_length, 3, 8], dtype=np.float32))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--occuppath", type=str, dest="occup_path",
                        required=True, help="Path to directory of the images")
    parser.add_argument("--occlpath", type=str, dest="occl_path",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--roadpath", type=str, dest="road_path",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--linespath", type=str, dest="lines_path",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--combpath", type=str, dest="combined_mask_path",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--tfpath", type=str, dest="transformation_path",
                        default=None, help="Path to directory where images should be saved")
    parser.add_argument("--outpath", type=str, dest="out_path",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--seq", type=int, dest="seq_length",
                        default=100, help="Path to directory where images should be saved")
    parser.add_argument("--step", type=int, dest="step_size",
                        default=1, help="Path to directory where images should be saved")
    parser.add_argument("--imsize", type=int, dest="image_size",
                        default=128, help="Path to directory where images should be saved")
    parser.add_argument("--roadsuffix", type=str, dest="road_path_suffix",
                        default="_horizon_map_road.png", help="Path to directory where images should be saved")
    parser.add_argument("--occlsuffix", type=str, dest="occl_path_suffix",
                        default="_occupancy_occlusion.pgm", help="Path to directory where images should be saved")
    parser.add_argument("--linessuffix", type=str, dest="lines_path_suffix",
                        default="_horizon_map_lines.png", help="Path to directory where images should be saved")
    parser.add_argument("--combsuffix", type=str, dest="comb_path_suffix",
                        default="_combined_mask.ppm", help="Path to directory where images should be saved")
    parser.add_argument("--tfsuffix", type=str, dest="transformation_suffix",
                        default="_transformation.npz", help="Path to directory where images should be saved")
    parser.add_argument("--tfonly", type=str2bool, dest="transformation_only",
                        default=False, help="Path to directory where images should be saved")
    parser.add_argument("--outsuffix", type=str, dest="out_suffix",
                        default=None, help="Additional suffix of compressed files")
    parser.add_argument("--splitnumber", type=int, dest="split_number",
                        default=0, help="If dataset is splitted iterate this number over your splits. Default 0")
    parser.add_argument("--splitamount", type=int, dest="split_amount",
                        default=1, help="Amount of splits of the dataset. Default 1 = not splitted")

    args = parser.parse_args()
    main(**vars(args))
