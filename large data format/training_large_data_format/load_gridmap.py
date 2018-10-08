"""
Load functions for gridmap data used for the Motion-Content Network.
This file contains three different functions for loading a gridmap as input for a Motion-Content Network implementation like mcnet_deep_tracking.py.
The output of every function includes the loaded frame sequence and the difference between all successive frames.

Uses and is inspired by https://github.com/rubenvillegas/iclr2017mcnet/blob/master/src/utils.py
MIT License, downloaded 07/01/2017
"""

import numpy as np
from utils import *     # see https://github.com/rubenvillegas/iclr2017mcnet/blob/master/src/utils.py - MIT License, downloaded 07/01/2017


def load_gridmap_data(f_name, data_path, image_size, K, T):
    """
    Load function for RGB gridmaps.
    Image sequences are loaded from a compressed numpy file ".npz". The array must have a shape of [image_size, image_size, frame_count * 3].
    The last channel consists of the three RGB channel for every image. This is why the first frame is saved at the position [:,:,0:3], the
    second at [:,:,3:6] and so on. All channels must be normalized between 1 and -1.

    Based on the loaded file the frame sequence is loaded in a [image_size, image_size, frame_count, 3] array for further processing. In addition the
    difference between each next frame is calculated and returned (for details see 'Returns').

    Args:
        f_name - Name of compressed numpy file which should be loaded
        image_size - Size of images in this file
        frame_count - Number of images which should be loaded

    Returns:
        seq - sequence of frames in an array of shape [image_size, image_size, frame_count, 3]
        diff - difference between each next frame. It is calculated by transforming normalized frames back to values between 0 and 1
                and subtract frame at t-1 from frame at t for diff[:,:,t,:].
    """
    f_name = f_name.split('\n', -1)[0]
    seq = np.load(f_name)['arr_0']

    diff = np.zeros((image_size, image_size, K - 1, 3), dtype="float32")
    for t in range(1, K):
        prev = inverse_transform(seq[:, :, t - 1])
        next = inverse_transform(seq[:, :, t])
        diff[:, :, t - 1] = next.astype("float32") - prev.astype("float32")
    return seq, diff


def load_gridmap_with_occlusion_data(f_name, image_size, frame_count):
    """
    Load function for RGB gridmaps with occlusion map.
    This function is based on 'load_gridmap_data' but extends it by loading an additional occlusion map channel. So the array must have a shape of
    [image_size, image_size, frame_count * 4]. The first frame RGB image is saved at the position [:,:,0:3] and its occlusion map at [:,:,3:4] and so on.
    All channels, even the occlusion map, must be normalized between 1 and -1. For the occlusion map 1 means visible and -1 occcluded.

    Args:
        f_name - Name of compressed numpy file which should be loaded
        image_size - Size of images in this file
        frame_count - Number of images which should be loaded

    Returns:
        seq - sequence of frames in an array of shape [image_size, image_size, frame_count, 4]
        diff - difference between each next frame. It is calculated by transforming normalized frames back to values between 0 and 1
                and subtract frame at timestep t-1 from frame at timestep t for diff[:,:,t,:]. The occlusion map as fourth channel is not subtracted but
                taken from frame at timestep t.
    """
    f_name = f_name.split('\n', -1)[0]
    seq = np.load(f_name)['arr_0']
    if seq.shape[2] > frame_count:
        seq = seq[:, :, :frame_count * 4]
    seq = np.stack(np.split(seq, frame_count, axis=2), axis=2)
    diff = np.zeros((image_size, image_size, frame_count, 4), dtype="float32")
    for t in range(1, frame_count):
        # Transformation of normalized frames to range of 0 to 1
        prev = inverse_transform(seq[:, :, t - 1, :3])
        next = inverse_transform(seq[:, :, t, :3])
        # Subtract frames and pass occlusion map
        diff[:, :, t, :3] = next.astype("float32") - prev.astype("float32")
        diff[:, :, t, 3:] = seq[:, :, t, 3:]
    return seq, diff


def load_gridmap_occupancy_data(f_name, image_size, frame_count):
    """
    Load function for gridmaps only consisting of occupancy and occlusion map.
    Works like load_gridmap_with_occlusion_data, but expects instead of a RGB image a single occupancy channel for each frame. So the array
    must have a shape of [image_size, image_size, frame_count * 2].

    Args:
        f_name - Name of compressed numpy file which should be loaded
        image_size - Size of images in this file
        frame_count - Number of images which should be loaded

    Returns:
        seq - sequence of frames in an array of shape [image_size, image_size, frame_count, 2]
        diff - difference between each next frame (for details see 'load_gridmap_with_occlusion_data').
    """
    f_name = f_name.split('\n', -1)[0]
    seq = np.load(f_name)['arr_0']
    seq = np.stack(np.split(seq, (K + T) * iterations, axis=2), axis=2)
    diff = np.zeros((image_size, image_size, iterations *
                     (K + T), 2), dtype="float32")
    for t in range(1, iterations * (K + T)):
        prev = inverse_transform(seq[:, :, t - 1, :1])
        next = inverse_transform(seq[:, :, t, :1])
        diff[:, :, t, :1] = next.astype("float32") - prev.astype("float32")
        diff[:, :, t, 1:] = seq[:, :, t, 1:]
    return seq, diff

def load_gridmap_content_motion_data(f_name, image_size, frame_count):
    """
    Load function for gridmaps consisting of occupancy as content, velocity as motion and occlusion as mask.
    Works like load_gridmap_with_occlusion_data, but expects instead of a RGB image an occupancy and velocity channel for each frame. So the array
    must have a shape of [image_size, image_size, frame_count * 3].

    Args:
        f_name - Name of compressed numpy file which should be loaded
        image_size - Size of images in this file
        frame_count - Number of images which should be loaded

    Returns:
        seq - sequence of frames in an array of shape [image_size, image_size, frame_count, 3]
        content - occupancy with occlusion map of whole sequence
        motion - velocity with occlusion map of whole sequence
    """
    f_name = f_name.split('\n', -1)[0]
    seq = np.load(f_name)['arr_0']
    # Split [image_size, image_size, frame_count * 3] to [image_size, image_size, frame_count, 3]
    seq = np.stack(np.split(seq, frame_count, axis=2), axis=2)
    content = seq[:,:,:,::2]  # Occupancy channel 0 and occlusion channel 2
    motion = seq[:,:,:,1:]      # Velocity channel 1 and occlusion channel 2
    return seq, content, motion

def load_gridmap_motion_maps(f_name, image_size, frame_count):
    """
    Load function for gridmaps consisting of occupancy, occlusion and motion maps.
    Works like load_gridmap_with_occlusion_data, but expects instead of a RGB image an one-channel occupancy map for each frame.
    In addition the first for channels contain the motion maps (0:3 - motion_orientation, 3:4 - motion_intensity). So the array
    must have a shape of [image_size, image_size, frame_count * 2 + 4].

    Args:
        f_name - Name of compressed numpy file which should be loaded
        image_size - Size of images in this file
        frame_count - Number of images which should be loaded

    Returns:
        seq - sequence of frames in an array of shape [image_size, image_size, frame_count, 2]
        motion_maps - 4 channel motion map (first three motion_orientation, other one motion_intensity)
    """
    f_name = f_name.split('\n', -1)[0]
    seq = np.load(f_name)['arr_0']
    motion_maps = seq[:,:,0:4]
    # Split [image_size, image_size, frame_count * 2] to [image_size, image_size, frame_count, 2]
    seq = np.stack(np.split(seq[:,:,4:frame_count*2+4], frame_count, axis=2), axis=2)
    return seq, motion_maps

def load_gridmap_onmove(f_name, image_size, frame_count, useCombinedMask=False):
    """
    Load function for gridmaps consisting of occupancy, occlusion, horizon map (road and lines) and combined mask of occlusion and map.
    Works like load_gridmap_motion_maps, but uses instead of the motion maps the horizon map of each time frame. For better understanding
    the map has to be 2 channel (road itself 1|-1 and lines 1|-1). In addition to select whether only objects on the road should be taken into
    the loss function or every object the last channel has to provide a map which combined the occlusion map with the road map.
    So the array must have a shape of [image_size, image_size, frame_count * 5].

    Args:
        f_name - Name of compressed numpy file which should be loaded
        image_size - Size of images in this file
        frame_count - Number of images which should be loaded
        useCombinedMask - determines whether the combined mask of road and occlusion map or only occlusion should be used

    Returns:
        target_seq - sequence of frames in an array of shape [image_size, image_size, frame_count, 2]
        input_seq - sequence of frames for the input. Includes occupancy and occlusion map [image_size, image_size, frame_count + 1, 2]
        maps - sequence of the horizon map. Channel 0 lines, channel 1 road. The shape is [image_size, image_size, frame_count, 2]
    """
    f_name = f_name.split('\n', -1)[0]
    #print("testing parallelism: "+f_name)
    seq = np.load(f_name)['arr_0']
    # Split [image_size, image_size, frame_count * 2] to [image_size, image_size, frame_count, 2]
    seq = np.stack(np.split(seq[:,:,:frame_count*5], frame_count, axis=2), axis=2)
    if image_size < seq.shape[0]:
        orig_size = seq.shape[0]
        start_index = (orig_size - image_size) // 2
        end_index = start_index + image_size
        seq = seq[start_index:end_index,start_index:end_index]

    input_seq = seq[:,:,:-1,0:2]
    maps = seq[:,:,:,2:4]
    if useCombinedMask:
        loss_mask = seq[:,:,1:,4:5]
        input_seq[:,:,:,0:1] = np.multiply((seq[:,:,:-1,4:5] + 1) // 2, (input_seq[:,:,:,0:1] + 1) // 2) * 2 - 1
    else:
        loss_mask = seq[:,:,1:,1:2]
        input_seq[:,:,:,0:1] = np.multiply((seq[:,:,:-1,4:5] + 1) // 2, (input_seq[:,:,:,0:1] + 1) // 2) * 2 - 1
        seq[:,:,1:,0:1] = np.multiply((seq[:,:,1:,0:1] + 1) // 2, (seq[:,:,1:,3:4] + 1) / 2) * 2 - 1
    target_seq = np.concatenate([seq[:,:,1:,0:1],loss_mask], axis=3)

    tf_name = f_name.split('.',-1)[0] + "_transformation.npz"
    # print tf_name
    try:
        tf_matrix = np.load(tf_name)['arr_0'][:frame_count-1]
        tf_matrix[:,:,2] = tf_matrix[:,:,2]
        tf_matrix[:,:,5] = - tf_matrix[:,:,5]
    except IOError:
        print("IOError for " + tf_name)
        tf_matrix = np.zeros([frame_count-1, 3, 8], dtype=np.float32)
        tf_matrix[:,:,0] = 1
        tf_matrix[:,:,4] = 1

    return target_seq, input_seq, maps, tf_matrix
