import os
import cv2
import sys
import time
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
import imageio
import math

import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
#from Tkinter.ttk import Frame, Button, Style

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

import sys
#change this to where repository is located
path_to_evalmodel = 'F:/selfdriving/large data format/training_large_data_format'
sys.path.insert(0, path_to_evalmodel)

from move_network_distributed_noGPU import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
import ../preprocessing_situ_all_data as pp

thresh = 96

#global variable init
return_clips = []
return_transformation = []
occupancy_buffer = []
transformation_buffer = []
occlusion_buffer = []
rgb_buffer = []
depth_buffer = []
segmentation_buffer = []
data_size = []
direction_buffer = []
sess = -1
model = -1
image_size = -1
data_w = -1
data_h = -1
canvas = -1

#following functions are from the preprocessing script
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

def trigClip(x,fn):
    return fn(np.clip(x,-1,1))

def inverseTransformationMatrix(nextTf):
    mat = np.zeros([3,2,3], dtype=np.float32)
    mat[:,0,:] = nextTf[:,0:3]
    mat[:,1,:] = nextTf[:,3:6]
    matFull = np.clip(mat[0,:],-1,1)
    #mean theta extracted from matrix
    theta = -(math.asin(-matFull[0,1])+math.asin(matFull[1,0]))/2
    imsize = 96 // 2
    pixel_diff_y = matFull[1,2] * ((imsize - 1) / 2.0)
    pixel_diff_x = matFull[0,2] * ((imsize - 1) / 2.0)
    py = pixel_diff_y / math.cos(theta)
    px = pixel_diff_x / math.sin(theta)

    if np.isinf(px) or np.isnan(px):
        pixel_diff = py
    elif np.isinf(py) or np.isnan(py):
        pixel_diff = px
    else:
        pixel_diff = (px+py)/2
    pixel_size = 45.6 * 1.0 / imsize
    period_duration = 1.0 / 24
    vel = pixel_diff * pixel_size / period_duration
    yaw_rate = math.degrees(theta) / period_duration
    return vel, yaw_rate

def compressMoveMapDataset(occupancy_buffer, occlusion_buffer, transformation_buffer, seq_length,
                           transformation_only=False, split_number = 0, split_amount = 1):
    global return_clips
    global return_transformation
    max_occup_diff = 0
    min_occup_diff = 100000
    mean_occup_diff = 0.0
    all_clips = [create_default_element(image_size, seq_length, 5) for i in range(seq_length)]
    all_transformation = [np.zeros([seq_length, 3, 8], dtype=np.float32) for i in range(seq_length)]

    #step_offset = int(round(1.0 * step_size / split_amount * split_number))

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

def preprocessing(img, rgb, depth, segmentation, yaw_rate, speed):
    global occupancy_buffer, occlusion_buffer, transformation_buffer, rgb_buffer, depth_buffer
    global segmentation_buffer, data_size, direction_buffer, return_clips, return_transformation
    seq_length = K + 1
    
    #find the direction (left,right,straight)
    if yaw_rate < -0.5: #going left
        direction_buffer.append([1,1])
    elif yaw_rate > 0.5: #going right
        direction_buffer.append([0,1])
    else: #going straight
        direction_buffer.append([0,0])
    new_size = tuple(t//8 for t in rgb.shape[:-1]) # from 1920 x 640 to 240 x 80
    data_size = [new_size[1], new_size[0]]
    new_size = tuple(data_size)
    rgb_buffer.append(pp.transform_input(rgb, new_size, False))
    depth_buffer.append(pp.transform_input(depth, new_size, True))
    segmentation_buffer.append(pp.transform_input(segmentation, new_size, False))
    occupancy_img = pp.cropAndResizeImage(img)
    occupancy_array = np.array(occupancy_img)
    if len(occupancy_array.shape) == 3:
        occupancy_array = np.mean(occupancy_array, axis=2)
    occlusion_array = pp.createOcclusionMap(occupancy_array)
    occupancy_mask = pp.createOccupancyMask(occupancy_img, occlusion_array, thresh)
    transformation_matrix = pp.calcImageTranslation(occupancy_array, yaw_rate, speed)
    occupancy_buffer.append(occupancy_mask)
    occlusion_buffer.append(occlusion_array)
    transformation_buffer.append(transformation_matrix)
    if len(occupancy_buffer) >= K+1:
        return_clips = []
        return_transformation = []
        compressMoveMapDataset(occupancy_buffer, occlusion_buffer, transformation_buffer, seq_length)
        return_camera_rgb = np.array(rgb_buffer)
        return_camera_segmentation = np.array(segmentation_buffer)
        return_camera_depth = np.expand_dims(np.array(depth_buffer),axis=-1)
        return_direction = np.array(direction_buffer).astype(np.uint8)
        tmpClips = np.stack(np.split(np.array(return_clips[0])[:,:,:seq_length*5], seq_length, axis=2), axis=2)
        input_seq = tmpClips[:,:,:-1,0:2]
        maps = tmpClips[:,:,:,2:4]
        input_seq[:,:,:,0:1] = np.multiply((tmpClips[:,:,:-1,4:5] + 1) // 2, (input_seq[:,:,:,0:1] + 1) // 2) * 2 - 1
        tf_matrix = np.array(return_transformation)[0]#[:seq_length-1]
        tf_matrix = tf_matrix[:seq_length-1]
        tf_matrix[:,:,2] = tf_matrix[:,:,2]
        tf_matrix[:,:,5] = - tf_matrix[:,:,5]
        del occupancy_buffer[0]
        del occlusion_buffer[0]
        del transformation_buffer[0]
        del rgb_buffer[0]
        del depth_buffer[0]
        del segmentation_buffer[0]
        del direction_buffer[0]
        return True, np.expand_dims(input_seq, axis=0), np.expand_dims(maps, axis=0), np.expand_dims(tf_matrix, axis=0), np.expand_dims(return_camera_rgb[1:], axis=0), np.expand_dims(return_camera_segmentation[1:], axis=0), np.expand_dims(return_camera_depth[1:], axis=0), np.expand_dims(return_direction[1:], axis=0)
    
    return False, -1, -1, -1, -1, -1, -1, -1


def init(checkpoint_dir_loc, prefix, image_size_i=96, data_w_i=240, data_h_i=80, K_i=9, T_i=10, seq_steps=1, useDenseBlock=False, samples=1):
    global K, T, sess, model, image_size, data_h, data_w, canvas
    root = tk.Tk()
    canvas = tk.Canvas(root, width=(data_w_i*3+data_h_i), height=(data_h_i*T_i), bd=0, highlightthickness=0)
    canvas.pack()
    data_w = data_w_i
    data_h = data_h_i
    image_size = image_size_i
    gpu = np.arange(1)
    K = K_i
    T = T_i
    print("Setup variables...")
    datasze_tf = np.zeros(2)
    datasze_tf[0] = data_w
    datasze_tf[1] = data_h
    imgsze_tf = image_size
    seqlen_tf = seq_steps
    K_tf = K_i
    T_tf = T_i
    fc_tf = K #we have K frames, we predict next T
    assert(seq_steps <= seqlen_tf)
    assert(K <= K_tf)
    assert(T <= T_tf)
    assert(seqlen_tf == 1)

    #first dim = batch_size, set to 1, needed for compatibility with model
    input_batch_shape = [1, imgsze_tf, imgsze_tf, seqlen_tf*(K_tf), 2]
    maps_batch_shape = [1, imgsze_tf, imgsze_tf, seqlen_tf*(K_tf)+1, 2]
    transformation_batch_shape = [1, seqlen_tf*(K_tf),3,8]
    rgb_batch_shape = [1, fc_tf,datasze_tf[1],datasze_tf[0],3]
    segmentation_batch_shape = [1, fc_tf,datasze_tf[1],datasze_tf[0],3]
    depth_batch_shape = [1, fc_tf,datasze_tf[1],datasze_tf[0],1]
    direction_batch_shape = [1, fc_tf,2]

    graph = tf.Graph()
    with graph.as_default():
        checkpoint_dir = checkpoint_dir_loc + prefix + "/"
        best_model = None  # will pick last model

        # initialize model
        model = MCNET(image_size=[image_size, image_size], data_size=[data_h, data_w], batch_size=1, K=K,
                  T=T, c_dim=1, checkpoint_dir=checkpoint_dir,
                  iterations=seq_steps, useSELU=True, motion_map_dims=2,
                  showFutureMaps=False, useDenseBlock=useDenseBlock, samples=samples)

        # Setup model (for details see training_large_data_format folder)
        model.pred_occlusion_map = tf.ones(model.occlusion_shape, dtype=tf.float32, name='Pred_Occlusion_Map') * model.predOcclValue
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            with tf.device("/gpu:%d" % gpu[0]):

                #fetch input
                model.input_batch = tf.placeholder(tf.float32, shape=input_batch_shape,name='input_batch')
                model.map_batch = tf.placeholder(tf.float32, shape=maps_batch_shape,name='map_batch')
                model.transformation_batch = tf.placeholder(tf.float32, shape=transformation_batch_shape,name='transformation_batch')
                model.rgb_cam = tf.placeholder(tf.float32, shape=rgb_batch_shape,name='rgb_cam')
                model.seg_cam = tf.placeholder(tf.float32, shape=segmentation_batch_shape,name='seg_cam')
                model.dep_cam = tf.placeholder(tf.float32, shape=depth_batch_shape,name='dep_cam')
                model.direction = tf.placeholder(tf.uint8, shape=direction_batch_shape,name='direction')

                # Construct the model
                pred, _, _, rgb_pred, seg_pred, dep_pred, _, _, trans_pred, dir_pred = model.forward(model.input_batch, model.map_batch, model.transformation_batch, model.rgb_cam, model.seg_cam, model.dep_cam, model.direction, 0)

                model.G = tf.stack(axis=3, values=pred)
                model.trans_pred = trans_pred
                model.rgb_pred = tf.stack(rgb_pred,1)
                model.seg_pred = tf.stack(seg_pred,1)
                model.dep_pred = tf.stack(dep_pred,1)

                tf.get_variable_scope().reuse_variables()

    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)

    with graph.as_default():
        
        model.saver = tf.train.Saver()
        
        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options))

        sess.run(init)

        bool, ckpt = model.load(sess, checkpoint_dir, best_model)
        print(checkpoint_dir)
        if bool:
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed... exitting")
            print(" [!] Checkpoint file is: "+str(ckpt))
            if ckpt != None:
                print(ckpt.model_checkpoint_path)
            return

def eval(input_gridmap, rgb, dep, seg, yaw_rate, speed):
    # preprocess input
    ppTime = time.time()
    ready, gridmap, gm_map, trans_matrix, rgb, seg, dep, dir_vehicle = preprocessing(input_gridmap, rgb, dep, seg, yaw_rate, speed)
    ppTime = time.time() - ppTime
    if ready:
        evTime = time.time()
        samples, rgb_pred, seg_pred, dep_pred, tfmat = sess.run([model.G, model.rgb_pred, model.seg_pred, model.dep_pred, model.trans_pred], 
                                                         feed_dict={model.input_batch: gridmap,
                                                                    model.map_batch: gm_map,
                                                                    model.transformation_batch: trans_matrix,
                                                                    model.rgb_cam: rgb,
                                                                    model.seg_cam: seg,
                                                                    model.dep_cam: dep,
                                                                    model.direction: dir_vehicle})

        
        evTime = time.time() - evTime
        imTime = time.time()
        
        samples_seq_step = (samples[0, :, :,:].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
        samples_seq_step = np.tile(samples_seq_step, [1,1,1,3])

        curr_frame = []

        rgb_pred = (np.clip(rgb_pred,-1,1)+1)/2 * 255
        seg_pred = (seg_pred+1)/2 * 255
        dep_pred = (dep_pred+1)/2 * 255

        for seq_step in range(T):
            pred = np.squeeze(samples_seq_step[K+seq_step])
            pred = (pred * 255).astype("uint8")
            curr_frame.append(np.concatenate([
                np.asarray(Image.fromarray(pred).resize((data_h,data_h),Image.ANTIALIAS)),
                      rgb_pred[0,K+seq_step,:,:,:],
                      seg_pred[0,K+seq_step,:,:,:],
                      np.tile(dep_pred[0,K+seq_step,:,:,:],[1,1,3])],1))

        npImg = np.uint8(np.concatenate(curr_frame,0))
        
        im = Image.fromarray(npImg)
        imTime = time.time() - imTime
        tkTime = time.time()
        image1 = ImageTk.PhotoImage(im)
        canvas.create_image(0, 0, image=image1, anchor="nw")
        canvas.update()
        tkTime = time.time() - tkTime
        return ppTime, evTime, imTime, tkTime
    
    return ppTime, 0, 0, 0
