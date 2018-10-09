import argparse
import logging
import random
import time
import math
import os
import numpy as np
import PIL
from tqdm import tqdm
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps
CUDA_VISIBLE_DEVICES = []

dest_path = None
prefix = None
samples_per_record = 20
K = 9
T = 10
prescale=337
crop_size=96
image_size=96
occup_steps=1
seq_length=20
step_size=5
assert((K+T)<=seq_length)
seq_steps=seq_length//(K+T)
#frame_composing=2 not supported for now, need to change code in calcImageTranslation first
thresh = 96 #Threshold for mean of pixels over channels (between 0 and 255)

occupancy_buffer = []
transformation_buffer = []
occlusion_buffer = []
return_clips = []
return_transformation = []
rgb_buffer = []
depth_buffer = []
segmentation_buffer = []
return_camera_rgb = []
return_camera_segmentation = []
return_camera_depth = []
idnr = 0
data_size = []
direction_buffer = []
return_direction = []
episode = 0

def set_dest_path(path, _samples_per_record, _K, _T, _image_size, _seq_length, _step_size):
    global dest_path
    global samples_per_record
    global K, T, prescale, crop_size, image_size, seq_length
    global step_size, thresh
    dest_path = path
    if not os.path.exists(path):
        os.makedirs(path)
        
    samples_per_record = _samples_per_record
    K = _K
    T = _T
    prescale = _prescale
    crop_size = _image_size
    image_size = _image_size
    seq_length = _seq_length
    step_size = _step_size

def update_episode(pfx):
    global prefix
    prefix = pfx

def main(img, rgb, depth, segmentation, yaw_rate, speed):
    global occupancy_buffer
    global occlusion_buffer
    global transformation_buffer
    global return_clips
    global return_transformation
    global rgb_buffer
    global depth_buffer
    global segmentation_buffer
    global return_camera_rgb
    global return_camera_segmentation
    global return_camera_depth
    global idnr
    global data_size
    global direction_buffer
    global return_direction
    if yaw_rate < -0.5: #going left
        direction_buffer.append([1,1])
    elif yaw_rate > 0.5: #going right
        direction_buffer.append([0,1])
    else: #going straight
        direction_buffer.append([0,0])
    new_size = tuple(t//8 for t in rgb.shape[:-1]) # from 1920 x 640 to 240 x 80
    data_size = [new_size[1], new_size[0]]
    new_size = tuple(data_size)
    rgb_buffer.append(transform_input(rgb, new_size, False))
    depth_buffer.append(transform_input(depth, new_size, True))
    segmentation_buffer.append(transform_input(segmentation, new_size, False))
    occupancy_img = cropAndResizeImage(img)
    occupancy_array = np.array(occupancy_img)
    if len(occupancy_array.shape) == 3:
        occupancy_array = np.mean(occupancy_array, axis=2)
    occlusion_array = createOcclusionMap(occupancy_array)
    occupancy_mask = createOccupancyMask(occupancy_img, occlusion_array, thresh)
    #occluded_array = createOcclusionImages(occupancy_mask, occlusion_array)
    transformation_matrix = calcImageTranslation(occupancy_array, yaw_rate, speed)
    occupancy_buffer.append(occupancy_mask)
    occlusion_buffer.append(occlusion_array)
    transformation_buffer.append(transformation_matrix)
    #for testing
    #return occupancy_mask, occlusion_array#, transformation_matrix
    if len(occupancy_buffer) >= seq_length:
        compressMoveMapDataset(occupancy_buffer, occlusion_buffer, transformation_buffer)
        return_camera_rgb.append(np.array(rgb_buffer))
        return_camera_segmentation.append(np.array(segmentation_buffer))
        return_camera_depth.append(np.array(depth_buffer))
        return_direction.append(direction_buffer)
        if len(return_clips) >= samples_per_record:
            #return return_clips
            convert_tfrecord(return_clips, np.array(return_camera_rgb), np.array(return_camera_segmentation), np.array(return_camera_depth), np.array(return_direction).astype(np.uint8))
            idnr += 1
            return_clips = []
            return_transformation = []
            return_camera_rgb = []
            return_camera_segmentation = []
            return_camera_depth = []
        del occupancy_buffer[:step_size]
        del occlusion_buffer[:step_size]
        del transformation_buffer[:step_size]
        del rgb_buffer[:step_size]
        del depth_buffer[:step_size]
        del segmentation_buffer[:step_size]
        del direction_buffer[:step_size]


def transform_input(responses, size, convert=True):
    responses = Image.fromarray(np.uint8(responses))
    if convert:
        return (np.array(responses.resize(size, Image.ANTIALIAS).convert('L')) / 127 - 1).astype(np.float32)
    else:
        return (np.array(responses.resize(size, Image.ANTIALIAS)) / 127 - 1).astype(np.float32)

def cropAndResizeImage(img):
    if prescale > 0:
        img.thumbnail((prescale, prescale), PIL.Image.ANTIALIAS)
    crop_img = ImageOps.fit(img, [crop_size, crop_size])
    scaled_img = crop_img.resize((image_size, image_size), PIL.Image.ANTIALIAS)
    return scaled_img

two_pi_f = 2 * math.pi
angular_res_rad_f = two_pi_f/900.0
radial_res_meter_f = 0.2
radial_limit_meter_f = 73
grid_cell_size = 0.4
grid_cell_size_inv_f = 1 / grid_cell_size
occ_thresh_f = 96

def createOcclusionMap(gridmap, max_occluded_steps=1):
    num_cells_per_edge_ui = gridmap.shape[0]
    num_cells_per_edge_half_f = gridmap.shape[0] // 2 - 1

    occlusion_map = np.ones(gridmap.shape, dtype=np.float32)    # 0 - occluded, 1 - non occluded/visible
    start_time = time.time()

    # Angle array captures 0 to 360 degree in radians to simulate the lidar beams
    angle_array = np.arange(0,two_pi_f,angular_res_rad_f)
    # Radial array captures 0 to max distance of detection to iterate over the distance to the ego vehicle
    radial_array = np.arange(0, radial_limit_meter_f, radial_res_meter_f)
    # For performance: repeat both arrays up to the shape of the other one to do faster matrix operations
    angle_array = np.stack([angle_array]*radial_array.shape[0], axis=1)
    radial_array = np.stack([radial_array]*angle_array.shape[0], axis=0)

    # x,y grid contains all x,y-Coordinates which correlate to the given angle and radius
    xy_grid = np.empty((angle_array.shape[0], radial_array.shape[1], 2), dtype=int)
    xy_grid[:,:,0] = grid_cell_size_inv_f * np.multiply(np.cos(angle_array), radial_array) + num_cells_per_edge_half_f # 0 - x
    xy_grid[:,:,1] = grid_cell_size_inv_f * np.multiply(np.sin(angle_array), radial_array) + num_cells_per_edge_half_f # 1 - y
    xy_grid = np.clip(xy_grid, 0, int(num_cells_per_edge_ui-1))

    occluded_steps = np.zeros((xy_grid.shape[0]), dtype=np.int32)
    is_occluded_array = np.zeros((xy_grid.shape[0]), dtype=np.bool)
    occlusion_wo_occup = np.ones((xy_grid.shape[0]), dtype=np.bool)
    position_array = np.zeros((xy_grid.shape[0], 2), dtype=int)

    for radial_index in range(xy_grid.shape[1]):
        x_i = xy_grid[:, radial_index, 0]
        y_i = xy_grid[:, radial_index, 1]

        occ_f = gridmap[y_i, x_i]
        is_occupied = (occ_f < occ_thresh_f)
        is_changed = is_occupied * (1 - is_occluded_array)
        position_array[:,0] = position_array[:,0] * (1 - is_changed) + x_i * (is_changed)
        position_array[:,1] = position_array[:,1] * (1 - is_changed) + y_i * (is_changed)
        is_occluded_array = is_occluded_array + is_occupied
        is_first_pixel = (np.absolute(position_array[:,0] - x_i) <= max_occluded_steps) * (np.absolute(position_array[:,1] - y_i) <= max_occluded_steps) * is_occupied

        occlusion_map[y_i, x_i] = occlusion_map[y_i, x_i] * (1 - (is_occluded_array * (1 - is_first_pixel)))

    return occlusion_map

def createOccupancyMask(occupancy_img, occlusion_array, thresh):
    img = np.array(occupancy_img).astype(np.float32)
    if len(img.shape) == 3:
        img = np.mean(img, axis=2)
    img = (img < thresh) * 1.0
    img = draw_ego_vehicle(img)
    occupancy_mask = np.multiply(img, occlusion_array)
    return occupancy_mask

def draw_ego_vehicle(img):
    image_size = img.shape[0]
    VEHICLE_WIDTH = image_size * 4 // 96
    VEHICLE_HEIGHT = image_size * 7 // 96
    x_start = image_size // 2 - VEHICLE_WIDTH // 2
    x_end = x_start + VEHICLE_WIDTH
    y_start = image_size // 2
    y_end = y_start + VEHICLE_HEIGHT
    img[y_start:y_end, x_start:x_end] = 1
    return img

def createOcclusionImages(img, occlusion_map):
    if len(img.shape) == 2:
        img = np.reshape(img, [img.shape[0], img.shape[1], 1])
    if img.shape[2] == 1:
        img = np.concatenate([img]*3, axis=2)
        img = img + (255 - img) * 0.5
    occlusion_map = np.stack([occlusion_map] * 3, axis=2)
    occluded_img = np.multiply(img, occlusion_map)
    return occluded_img

def calcImageTranslation(img, yaw_rate, vel, frame_rate=24, combined_frames=1, gridmap_size = 45.6):
    img = np.stack([img] * 3, axis=2)
    assert(img.shape[2] == 3)
    transform_matrix = np.zeros([3, 2, 3])
    for i in range(transform_matrix.shape[0]):
        if combined_frames == 1 or (len(frames) - frame_index <= combined_frames):
            theta, dx, dy = get_transformation_parameter(img.shape[0] // (2 ** i), gridmap_size, frame_rate, vel, yaw_rate)
        else: #this part needed for frame_composing > 1, currently not supported, needs list of imgs & odometry
            vels = []
            yaw_rates = []
            for comb_index in range(combined_frames):
                vels.append(odometry_dict[get_frame_number(frames[frame_index + comb_index])]['vel'])
                yaw_rates.append(odometry_dict[get_frame_number(frames[frame_index + comb_index])]['yaw_rate'])
            theta, dx, dy = get_combined_transformation_parameter(img.shape[0] // (2 ** i), gridmap_size, frame_rate, vels,yaw_rates)

        transform_matrix[i, :] = get_STM_matrix(img.shape[0] // (2 ** i), theta, dx, dy)
    return transform_matrix

def get_transformation_parameter(imsize, gridmap_size, frame_rate, vel, yaw_rate):
    period_duration = 1.0 / frame_rate
    yaw_diff = math.radians(yaw_rate * period_duration)
    pixel_size = gridmap_size * 1.0 / imsize    # [m]
    pixel_diff = vel * period_duration * 1.0 / pixel_size
    pixel_diff_y = math.cos(yaw_diff) * pixel_diff
    pixel_diff_x = math.sin(yaw_diff) * pixel_diff
    return yaw_diff, pixel_diff_x, pixel_diff_y

def get_STM_matrix(imsize, theta, dx, dy):
    theta = -theta
    a11 = math.cos(theta)
    a12 = -math.sin(theta)
    a13 = dx / ((imsize - 1) / 2.0)
    a21 = math.sin(theta)
    a22 = math.cos(theta)
    a23 = dy / ((imsize - 1) / 2.0)
    M = np.array([[a11, a12, a13], [a21, a22, a23]])
    return M

def create_default_element(image_size, seq_length, channel_size):
    element = np.zeros([image_size, image_size, seq_length * channel_size], dtype=np.float32)
    return element

def read_frame(occup_map, occlusion_map):
    #dummy values as long as no underlying road can be extracted from carla - backward comp. with NN model
    lines_map = np.ones(shape=(occup_map.shape)) * 255
    road_map = np.ones(shape=(occup_map.shape)) * 255
    if len(occlusion_map.shape) == 3:
        occlusion_map = np.mean(occlusion_map, axis=2)
    return np.stack([occup_map*255, occlusion_map*255, lines_map, road_map, occlusion_map*255], axis=2)

def read_matrix(matrix):
    tf_matrix = np.zeros([3,8], dtype=np.float32)
    tf_matrix[:,0:3] = matrix[:,0,:]
    tf_matrix[:,3:6] = matrix[:,1,:]
    return tf_matrix

def normalize_frames(frames):
    new_frames = frames.astype(np.float32)
    new_frames //= (255 // 2)
    new_frames -= 1
    return new_frames

def get_occupancy_diff(clip):
    occup_clip = np.multiply((clip[:,:,::5] + 1)/2.0, (clip[:,:,4::5] + 1)/2.0)
    occup_diff = occup_clip[:,:,:-1] - occup_clip[:,:,1:]
    occup_diff = np.absolute(occup_diff)
    return np.sum(occup_diff) * 1.0 / occup_diff.shape[2]

def compressMoveMapDataset(occupancy_buffer, occlusion_buffer, transformation_buffer,
                           transformation_only=False, split_number = 0, split_amount = 1):
    global return_clips
    global return_transformation
    max_occup_diff = 0
    min_occup_diff = 100000
    mean_occup_diff = 0.0
    all_clips = [create_default_element(image_size, seq_length, 5) for i in range(seq_length)]
    all_transformation = [np.zeros([seq_length, 3, 8], dtype=np.float32) for i in range(seq_length)]

    step_offset = int(round(1.0 * step_size / split_amount * split_number))

    for file_index in range(len(occupancy_buffer)):
        frame = read_frame(occupancy_buffer[file_index], occlusion_buffer[file_index])
        if transformation_only:
            frame = np.zeros([1,1,1])
        transform_matrix = read_matrix(transformation_buffer[file_index])

        channel_size = frame.shape[2]

        norm_frame = normalize_frames(frame)
        for clip_index in range(len(all_clips)):
            if not transformation_only:
                all_clips[clip_index][:, :, clip_index * channel_size: (clip_index + 1) * channel_size] = norm_frame
            all_transformation[clip_index][clip_index,:,:] = transform_matrix
        if file_index >= seq_length - 1:
            if not transformation_only:
                occup_diff = get_occupancy_diff(all_clips[-1])
                max_occup_diff = max(max_occup_diff, occup_diff)
                min_occup_diff = min(min_occup_diff, occup_diff)
                mean_occup_diff = mean_occup_diff + occup_diff
            if transformation_only or occup_diff >= 0: #was 6
                if not transformation_only:
                    return_clips.append(all_clips[-1])
                return_transformation.append(all_transformation[-1])
            else:
                print("Sequence not saved due to occupancy difference of " + str(occup_diff))

        del all_clips[-1]
        del all_transformation[-1]
        all_clips.insert(0, create_default_element(image_size, seq_length, channel_size))
        all_transformation.insert(0, np.zeros([seq_length, 3, 8], dtype=np.float32))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def load_gridmap_onmove_tfrecord(ind, size, frame_count, useCombinedMask=False):
    seq = return_clips[ind]
    seq = np.stack(np.split(seq[:,:,:frame_count*5], frame_count, axis=2), axis=2)
    if image_size < seq.shape[0]:
        orig_size = seq.shape[0]
        start_index = (orig_size - size) // 2
        end_index = start_index + size
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

    tf_matrix = return_transformation[ind][:frame_count-1]
    tf_matrix[:,:,2] = tf_matrix[:,:,2]
    tf_matrix[:,:,5] = - tf_matrix[:,:,5]

    return target_seq, input_seq, maps, tf_matrix

def convert_tfrecord(clips, return_camera_rgb, return_camera_segmentation, return_camera_depth, return_direction, useCombinedMask=False):
    samples = np.arange(samples_per_record)
    shapes = np.repeat(np.array([image_size]), 1, axis=0)
    sequence_steps = np.repeat(np.array([1 + seq_steps * (K + T)]), 1, axis=0)
    combLoss = np.repeat(useCombinedMask, 1, axis=0)
    strname = "imgsze="+str(image_size)+"_fc="+str(seq_length)+"_datasze="+str(data_size[0])+"x"+str(data_size[1])+"_seqlen="+str(seq_steps)+"_K="+str(K)+"_T="+str(T)+"_size="+str(samples_per_record)
    prefixname = prefix + '_' + str(idnr)
    filename = os.path.join(dest_path, prefixname + '_' + strname + '.tfrecord')
    with tf.python_io.TFRecordWriter(filename) as writer:
        for index in samples:
            for f, img_sze, seq, useCM in zip([index], shapes, sequence_steps, combLoss):
                target_seq, input_seq, maps, tf_matrix = load_gridmap_onmove_tfrecord(f, img_sze, seq, useCM)
            seq_batch = target_seq.tostring()
            input_batch = input_seq.tostring()
            map_batch = maps.tostring()
            transformation_batch = tf_matrix.tostring()
            rgb_batch = return_camera_rgb[index].tostring()
            segmentation_batch = return_camera_segmentation[index].tostring()
            depth_batch = return_camera_depth[index].tostring()
            direction_batch = return_direction[index].tostring()
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'input_seq': _bytes_feature(input_batch),
                        'target_seq': _bytes_feature(seq_batch),
                        'maps': _bytes_feature(map_batch),
                        'tf_matrix': _bytes_feature(transformation_batch),
                        'rgb': _bytes_feature(rgb_batch),
                        'segmentation': _bytes_feature(segmentation_batch),
                        'depth': _bytes_feature(depth_batch),
                        'direction': _bytes_feature(direction_batch)
                    }))
            writer.write(example.SerializeToString())
