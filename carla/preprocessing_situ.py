import argparse
import logging
import random
import time
import math
import os
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageOps

prescale=337
crop_size=96
image_size=96
occup_steps=1
seq_length=20
step_size=5
seq_steps=1
#frame_composing=2 not supported for now, need to change code in calcImageTranslation first
thresh = 96 #Threshold for mean of pixels over channels (between 0 and 255)
OVERWRITE_OCCLUSION="False"
OVERWRITE_OCCUPMASK="False"
OVERWRITE_OCCLIMGS="False"
BASEDIR='./preprocessed_dataset/'

image_buffer = []
transformation_buffer = []


def main(img, yaw_rate, speed):
    global image_buffer
    global transformation_buffer
    occupancy_img = cropAndResizeImage(img)
    occupancy_array = np.array(occupancy_img)
    occlusion_array = createOcclusionMap(occupancy_array)
    occupancy_mask = createOccupancyMask(occupancy_img, occlusion_array, thresh)
    occluded_array = createOcclusionImages(occupancy_mask, occlusion_array)
    transformation_matrix = calcImageTranslation(occupancy_array, yaw_rate, speed)
    image_buffer.append(occluded_array)
    transformation_buffer.append(transformation_matrix)
    #temporary for testing
    return occluded_array, transformation_matrix
    #if len(image_buffer) >= seq_length:
    #    compressMoveMapDataset()
    #    convert_tfrecord()
    #    del image_buffer[:step_size]
    #    del transformation_buffer[:step_size]
    
    
    
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

        # occluded_steps += np.multiply(np.ones(occluded_steps.shape, dtype=np.int32), is_occluded_array)
        # occluded_steps = np.multiply(is_occluded_array, )
        occ_f = gridmap[y_i, x_i]
        is_occupied = (occ_f < occ_thresh_f)
        is_changed = is_occupied * (1 - is_occluded_array)
        position_array[:,0] = position_array[:,0] * (1 - is_changed) + x_i * (is_changed)
        position_array[:,1] = position_array[:,1] * (1 - is_changed) + y_i * (is_changed)
        is_occluded_array = is_occluded_array + is_occupied 
        is_first_pixel = (np.absolute(position_array[:,0] - x_i) <= max_occluded_steps) * (np.absolute(position_array[:,1] - y_i) <= max_occluded_steps) * is_occupied
        # occlusion_wo_occup = (1 - is_occluded_array) + (is_occluded_array * occlusion_wo_occup * is_occupied)
        # occlusion_map[y_i, x_i] = occlusion_map[y_i, x_i] * (1 - (is_occluded_array * (1 - occlusion_wo_occup)))
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
    #occlusion_map = occlusion_path / 255.0 values probably already between 0-1
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