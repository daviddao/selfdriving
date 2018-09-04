import os
import cv2
import sys
import time
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
import imageio

import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
#from Tkinter.ttk import Frame, Button, Style
#import wand

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

import sys
sys.path.insert(0, 'F:/selfdriving/training_large_data_format')

from move_network_distributed_noGPU import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
import preprocessing_situ_all_data as pp

thresh = 96

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
    global occupancy_buffer
    global occlusion_buffer
    global transformation_buffer
    global rgb_buffer
    global depth_buffer
    global segmentation_buffer
    global data_size
    global direction_buffer
    global return_clips
    global return_transformation
    seq_length = K + 1
    
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
    #occluded_array = createOcclusionImages(occupancy_mask, occlusion_array)
    transformation_matrix = pp.calcImageTranslation(occupancy_array, yaw_rate, speed)
    occupancy_buffer.append(occupancy_mask)
    occlusion_buffer.append(occlusion_array)
    transformation_buffer.append(transformation_matrix)
    #for testing
    #return occupancy_mask, occlusion_array#, transformation_matrix
    if len(occupancy_buffer) >= K+1:
        return_clips = []
        return_transformation = []
        compressMoveMapDataset(occupancy_buffer, occlusion_buffer, transformation_buffer, seq_length)
        #return_clips = pp.return_clips
        #return_transformation = pp.return_transformation
        #pp.return_clips = []
        #pp.return_transformation = []
        return_camera_rgb = np.array(rgb_buffer)
        return_camera_segmentation = np.array(segmentation_buffer)
        return_camera_depth = np.expand_dims(np.array(depth_buffer),axis=-1)
        return_direction = np.array(direction_buffer).astype(np.uint8)
        #print(return_direction.shape)
        tmpClips = np.stack(np.split(np.array(return_clips[0])[:,:,:seq_length*5], seq_length, axis=2), axis=2)
        #print(tmpClips.shape)
        input_seq = tmpClips[:,:,:-1,0:2]
        maps = tmpClips[:,:,:,2:4]
        input_seq[:,:,:,0:1] = np.multiply((tmpClips[:,:,:-1,4:5] + 1) // 2, (input_seq[:,:,:,0:1] + 1) // 2) * 2 - 1
        tf_matrix = np.array(return_transformation)[0]#[:seq_length-1]
        tf_matrix = tf_matrix[:seq_length-1]
        #print(tf_matrix.shape)
        tf_matrix[:,:,2] = tf_matrix[:,:,2]
        tf_matrix[:,:,5] = - tf_matrix[:,:,5]
        del occupancy_buffer[0]
        del occlusion_buffer[0]
        del transformation_buffer[0]
        del rgb_buffer[0]
        del depth_buffer[0]
        del segmentation_buffer[0]
        del direction_buffer[0] #second return_clips arg probably wrong, should be maps but want to pass back zeros
        #RETURNS_CLIPS RETURNS 0 PROBABLY NEED TO HAVE LOCAL COMPRESSMOVEMAP FN HERE
        return True, np.expand_dims(input_seq, axis=0), np.expand_dims(maps, axis=0), np.expand_dims(tf_matrix, axis=0), np.expand_dims(return_camera_rgb[1:], axis=0), np.expand_dims(return_camera_segmentation[1:], axis=0), np.expand_dims(return_camera_depth[1:], axis=0), np.expand_dims(return_direction[1:], axis=0)
    
    return False, -1, -1, -1, -1, -1, -1, -1


def init(checkpoint_dir_loc, prefix, image_size_i=96, data_w_i=240, data_h_i=80, K_i=9, T_i=10, seq_steps=1, useDenseBlock=False, samples=1):
    global K
    global T
    global sess
    global model
    global image_size
    global data_h
    global data_w
    global canvas
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

    #first dim = batch_size, set to 1, needed for comp. with training model
    input_batch_shape = [1, imgsze_tf, imgsze_tf, seqlen_tf*(K_tf), 2]
    #seq_batch_shape = [1, imgsze_tf, imgsze_tf, seqlen_tf*(K_tf), 2]
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

        # Setup model (for details see mcnet_deep_tracking.py)
        model.pred_occlusion_map = tf.ones(model.occlusion_shape, dtype=tf.float32, name='Pred_Occlusion_Map') * model.predOcclValue
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            with tf.device("/gpu:%d" % gpu[0]):

                #fetch input
                #seq_batch = tf.placeholder(tf.float32, shape=seq_batch_shape,name='seq_batch')
                model.input_batch = tf.placeholder(tf.float32, shape=input_batch_shape,name='input_batch')
                model.map_batch = tf.placeholder(tf.float32, shape=maps_batch_shape,name='map_batch')
                model.transformation_batch = tf.placeholder(tf.float32, shape=transformation_batch_shape,name='transformation_batch')
                model.rgb_cam = tf.placeholder(tf.float32, shape=rgb_batch_shape,name='rgb_cam')
                model.seg_cam = tf.placeholder(tf.float32, shape=segmentation_batch_shape,name='seg_cam')
                model.dep_cam = tf.placeholder(tf.float32, shape=depth_batch_shape,name='dep_cam')
                model.direction = tf.placeholder(tf.uint8, shape=direction_batch_shape,name='direction')

                # Construct the model
                pred, _, _, rgb_pred, seg_pred, dep_pred, _, _, trans_pred, dir_pred = model.forward(model.input_batch, model.map_batch, model.transformation_batch, model.rgb_cam, model.seg_cam, model.dep_cam, model.direction, 0)

                #model.target = seq_batch
                #model.motion_map_tensor = map_batch
                model.G = tf.stack(axis=3, values=pred)
                #model.loss_occlusion_mask = (tf.tile(seq_batch[:, :, :, :, -1:], [1, 1, 1, 1, model.c_dim]) + 1) / 2.0
                #model.target_masked = model.mask_black(seq_batch[:, :, :, :, :model.c_dim], model.loss_occlusion_mask)
                #model.G_masked = model.mask_black(model.G, model.loss_occlusion_mask)

                #model.rgb = rgb_cam
                #model.seg = seg_cam
                #model.dep = dep_cam
                model.rgb_pred = tf.stack(rgb_pred,1)
                model.seg_pred = tf.stack(seg_pred,1)
                model.dep_pred = tf.stack(dep_pred,1)
                #model.rgb_diff = tf.reduce_mean(tf.square(model.rgb_pred - model.rgb))
                #model.seg_diff = tf.reduce_mean(tf.square(model.seg_pred - model.seg))
                #model.dep_diff = tf.reduce_mean(tf.square(model.dep_pred - model.dep))

                tf.get_variable_scope().reuse_variables()

    #deprecated!
    #if include_road:
    #    prefix = prefix + "_road_"
    #else:
    #    prefix = prefix + "_noRoad_"
    
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
    ready, gridmap, gm_map, trans_matrix, rgb, seg, dep, dir_vehicle = preprocessing(input_gridmap, rgb, dep, seg, yaw_rate, speed)
    
    if ready:
        samples, rgb_pred, seg_pred, dep_pred = sess.run([model.G, model.rgb_pred, model.seg_pred, model.dep_pred], feed_dict={model.input_batch: gridmap,
        model.map_batch: gm_map,
        model.transformation_batch: trans_matrix,
        model.rgb_cam: rgb,
        model.seg_cam: seg,
        model.dep_cam: dep,
        model.direction: dir_vehicle})

        samples_seq_step = (samples[0, :, :,:].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
        samples_seq_step = np.tile(samples_seq_step, [1,1,1,3])

        #pred_list = []
        #for t in range(T): #range(K+T)
        #    pred = np.squeeze(samples_seq_step[K+t])
        #    pred = (pred * 255).astype("uint8")
        #    pred_list.append(pred)

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
        #im = Image.new('RGB', npImg.shape[:2])
        #im.close()
        
        im = Image.fromarray(npImg)
        image1 = ImageTk.PhotoImage(im)
        #label = tk.Label(root, image=image1)
        #label.pack()
        canvas.create_image(0, 0, image=image1, anchor="nw")
        #canvas.update_idletasks()
        canvas.update()


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prefix", type=str, dest="prefix",
                        default="EncDense-BigLoop1-5_100kiter_GRIDMAP_MCNET_onmove_image_size=96_K=9_T=10_seqsteps=4_batch_size=4_alpha=1.001_beta=0.0_lr_G=0.0001_lr_D=0.0001_d_in=20_selu=True_comb=False_predV=-1", help="Prefix for log/snapshot")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=96, help="Pre-trained model")
    parser.add_argument("--K", type=int, dest="K",
                        default=9, help="Number of input images")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--useGAN", type=str2bool, dest="useGAN",
                        default=False, help="Model trained with GAN?")
    parser.add_argument("--useSharpen", type=str2bool, dest="useSharpen",
                        default=False, help="Model trained with sharpener?")
    parser.add_argument("--num_gpu", type=int, dest="num_gpu", required=True,
                        help="number of gpus")
    parser.add_argument("--data_path", type=str, dest="data_path", default="../preprocessing/preprocessed_dataset/BigLoopNew/",
                        help="Path where the test data is stored")
    parser.add_argument("--tfrecord", type=str, dest="tfrecord", default="all_in_one_new_shard_imgsze=96_seqlen=4_K=9_T=10_all",
                        help="Either folder name containing tfrecords or name of single tfrecord.")
    parser.add_argument("--road", type=str2bool, dest="include_road", default=False,
                        help="Should road be included?")
    parser.add_argument("--num_iters", type=int, dest="num_iters", default=10,
                        help="How many files should be checked?")
    parser.add_argument("--seq_steps", type=int, dest="seq_steps", default=1,
                        help="Number of iterations in model.")
    parser.add_argument("--denseBlock", type=str2bool, dest="useDenseBlock", default=True,
                        help="Use DenseBlock (dil_conv) or VAE-distr.")
    parser.add_argument("--samples", type=int, dest="samples", default=1,
                        help="if using VAE how often should be sampled?")
    parser.add_argument("--chckpt_loc", type=str, dest="checkpoint_dir_loc", default="./models/",
                        help="Location of model checkpoint file")
    parser.add_argument("--data_w", type=int, dest="data_w",
                        default=240, help="rgb/seg/depth image width size")
    parser.add_argument("--data_h", type=int, dest="data_h",
                        default=80, help="rgb/seg/depth image width size")

    args = parser.parse_args()
    main(**vars(args))
