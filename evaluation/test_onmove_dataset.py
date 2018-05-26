"""
Testscript for gridmap predictions of the train_onmove.py MCNet
Python 2

Inspired by https://github.com/rubenvillegas/iclr2017mcnet/blob/master/src/test_kth.py
MIT License, download 07/01/2017

Dependencies:
  - mcnet.py (for original network architecture)
  - mcnet_deep_tracking.py (for deep tracking)

Usage:
  (CUDA_VISIBLE_DEVICES=...) python test_onmove.py [--prefix PREFIX] [--image_size IMAGE_SIZE] [--K K] [--T T] [--gpu GPU] [--data_path DATA_PATH]

  Args:
    prefix - The prefix for the model which should be tested
    image_size - Size of images which should be used for testing (test images in data_path must have this shape)
    K - Determines how many images the network will get to see before prediction
    T - Determines how many images the network should predict
    gpu - GPU device id
    data_path - Path where the test data is stored. The parent directory must contain a file "test_data_list.txt" where all test file paths are listed

  Output:
    The predictions of the network are saved to ../results/images/Gridmap/PREFIX/
    In addition the compressed numpy array of PSNR errors on the test data is saved to ../results/quantitative/Gridmap/PREFIX/results_model=best_model.npz.
"""
import os
import cv2
import sys
import time
from skimage.measure import compare_ssim as ssim
from tqdm import tqdm
import imageio

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

from move_network_val import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw


def main(data_path, tfrecord, prefix, image_size, K, T, useGAN, useSharpen, num_gpu, include_road, num_iters, seq_steps, useDenseBlock, samples, checkpoint_dir_loc):
    """
    Main function for running the test. Arguments are passed from the command line

    Args:
      data_path - Path where the test data is stored. The parent directory must contain a file "test_data_list.txt" where all test file paths are listed
      prefix - The prefix for the model which should be tested
      image_size - Size of images which should be used for testing (test images in data_path must have this shape)
      K - Determines how many images the network will get to see before prediction
      T - Determines how many images the network should predict
      gpu - GPU device id
    """
    
    gpu = np.arange(num_gpu)
    #need at least 1 gpu to run code
    assert(num_gpu>=1 and len(gpu)==num_gpu)
    
    print("Setup dataset...")
    #def input_fn():
    imgsze_tf, seqlen_tf, K_tf, T_tf, nr_samples = parse_tfrecord_name(tfrecord)
    assert(image_size == imgsze_tf)
    assert(seq_steps <= seqlen_tf)
    assert(K <= K_tf)
    assert(T <= T_tf)
    def _parse_function(example_proto):
        keys_to_features = {'input_seq': tf.FixedLenFeature((), tf.string),
                            'target_seq': tf.FixedLenFeature((), tf.string),
                            'maps': tf.FixedLenFeature((), tf.string),
                            'tf_matrix': tf.FixedLenFeature((), tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)
        
        input_batch_shape = [imgsze_tf, imgsze_tf, seqlen_tf*(K_tf+T_tf), 2]
        seq_batch_shape = [imgsze_tf, imgsze_tf, seqlen_tf*(K_tf+T_tf), 2]
        maps_batch_shape = [imgsze_tf, imgsze_tf, seqlen_tf*(K_tf+T_tf)+1, 2]
        transformation_batch_shape = [seqlen_tf*(K_tf+T_tf),3,8]

        input_seq = tf.reshape(tf.decode_raw(parsed_features['input_seq'], tf.float32), input_batch_shape, name='reshape_input_seq')
        target_seq = tf.reshape(tf.decode_raw(parsed_features['target_seq'], tf.float32), seq_batch_shape, name='reshape_target_seq')
        maps = tf.reshape(tf.decode_raw(parsed_features['maps'], tf.float32), maps_batch_shape, name='reshape_maps')
        tf_matrix = tf.reshape(tf.decode_raw(parsed_features['tf_matrix'], tf.float32), transformation_batch_shape, name='reshape_tf_matrix')
        
        if (K+T)*seq_steps < input_seq.shape[2]:
            target_seq = target_seq[:,:,:(K+T)*seq_steps,:]
            input_seq = input_seq[:,:,:(K+T)*seq_steps,:]
            maps = maps[:,:,:(K+T)*seq_steps+1,:]
            tf_matrix = tf_matrix[:(K+T)*seq_steps,:,:]

        return target_seq, input_seq, maps, tf_matrix

    tfrecordsLoc = data_path + tfrecord
    #loading from directory containing sharded tfrecords or from one tfrecord
    if os.path.isdir(tfrecordsLoc):
        num_records = len(os.listdir(tfrecordsLoc))
        print("Loading from directory. " + str(num_records) + " tfRecords found.")
        files = tf.data.Dataset.list_files(tfrecordsLoc + "/" + "*.tfrecord").shuffle(num_records)
        dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(100),cycle_length=200)
    else:
        print("Loading from single tfRecord. " + str(nr_samples) + " entries in tfRecord.")
        dataset = tf.data.TFRecordDataset([tfrecordsLoc + '.tfrecord'])
    dataset = dataset.map(_parse_function, num_parallel_calls=64)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))
    dataset = dataset.prefetch(2)

    checkpoint_dir = checkpoint_dir_loc + prefix + "/"
    best_model = None  # will pick last model
    
    # initialize model
    model = MCNET(image_size=[image_size, image_size], batch_size=1, K=K,
              T=T, c_dim=1, checkpoint_dir=checkpoint_dir,
              iterations=seq_steps, useSELU=True, motion_map_dims=2,
              showFutureMaps=False, useGAN=useGAN, useSharpen=useSharpen, useDenseBlock=useDenseBlock, samples=samples)

    # Setup model (for details see mcnet_deep_tracking.py)
    model.pred_occlusion_map = tf.ones(model.occlusion_shape, dtype=tf.float32, name='Pred_Occlusion_Map') * model.predOcclValue
    iterator = dataset.make_initializable_iterator()
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        with tf.device("/gpu:%d" % gpu[0]):
            
            #fetch input
            seq_batch, input_batch, maps_batch, tf_batch = iterator.get_next()
            
            # Construct the model
            pred = model.forward(input_batch, maps_batch, tf_batch)
            
            model.target = seq_batch
            model.motion_map_tensor = maps_batch
            model.G = tf.stack(axis=3, values=pred)
            model.loss_occlusion_mask = (tf.tile(seq_batch[:, :, :, :, -1:], [1, 1, 1, 1, model.c_dim]) + 1) / 2.0
            model.target_masked = model.mask_black(seq_batch[:, :, :, :, :model.c_dim], model.loss_occlusion_mask)
            model.G_masked = model.mask_black(model.G, model.loss_occlusion_mask)
            
            if useGAN:
                start_frame = 0
                center_frame = model.d_input_frames // 2
                end_frame = model.d_input_frames
                gen_sequence = tf.concat(axis=3, values=[model.target_masked[
                                         :, :, :, start_frame:center_frame, :], model.G_masked[:, :, :, center_frame:end_frame, :]])
                gt_sequence = model.target_masked[:, :, :, start_frame:end_frame, :]
                good_data = tf.reshape(gt_sequence,
                                       [model.batch_size, model.image_size[0],
                                        model.image_size[1], -1])
                gen_data = tf.reshape(gen_sequence,
                                      [model.batch_size, model.image_size[0],
                                       model.image_size[1], -1])

                with tf.variable_scope("DIS", reuse=False):
                    model.D, model.D_logits = model.discriminator(good_data)

                with tf.variable_scope("DIS", reuse=True):
                    model.D_, model.D_logits_ = model.discriminator(gen_data)

                # Standard loss for real and fake (only for display and parameter
                # purpose, no loss trained on)

                model.d_loss_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=model.D_logits, labels=tf.ones_like(model.D)
                    )
                )
                model.d_loss_fake = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=model.D_logits_, labels=tf.zeros_like(model.D_)
                    )
                )
                
                
            if useGAN: #use GAN
                model.L_GAN = -tf.reduce_mean(model.D_)
                model.d_loss = model.d_loss_fake + model.d_loss_real
            else:
                model.d_loss_fake = tf.constant(0.0)
                model.d_loss_real = tf.constant(0.0)
                model.d_loss = tf.constant(0.0)
                model.L_GAN = tf.constant(0.0)
            
            tf.get_variable_scope().reuse_variables()

            
    
    if include_road:
        prefix = prefix + "_road_"
    else:
        prefix = prefix + "_noRoad_"
            
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options=gpu_options)) as sess:
        # Prepare model and directories
        tf.global_variables_initializer().run()
        
        model.saver = tf.train.Saver()

        quant_dir = "../results/quantitative/Gridmap/" + prefix + "/"
        save_path = quant_dir + "results_model=" + "best_model" + ".npz"
        if not exists(quant_dir):
            makedirs(quant_dir)

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

        vid_names = []
        psnr_err = np.zeros((0, T))  # Array for saving the PSNR error of all test sequences
        ssim_err = np.zeros((0, T))  # Array for saving the SSIM error of all test sequences

        sess.run(iterator.initializer)
        for i in tqdm(range(num_iters)): #hardcoded, should extract number of files from tfrecord

            samples, target_occ, motion_maps, occ_map = sess.run([model.G, model.target, model.motion_map_tensor, model.loss_occlusion_mask])

            # Save predictions and PSNR/SSIM error
            savedir = "../results/images/Gridmap/" + prefix + "/" + "save" + str(i)
                
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            cpsnr = np.zeros((K + T,))
            cssim = np.zeros((K + T,))
            
            maps_lines = (motion_maps[0, :, :, :, 0:1].swapaxes(0, 2).swapaxes(1, 2) + 1) // 2
            maps_road = (motion_maps[0, :, :, :, 1:2].swapaxes(0, 2).swapaxes(1, 2) + 1) // 2
            maps_lines = np.squeeze(np.tile(maps_lines, [1,1,1,3]))
            maps_road = np.squeeze(np.tile(maps_road, [1,1,1,3]))
            samples_seq_step = (samples[0, :, :,:].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
            sbatch = (target_occ[0, :, :,:, 0:1].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
            occ_map_step = occ_map[0, :, :,:].swapaxes(0, 2).swapaxes(1, 2)
            samples_seq_step = np.tile(samples_seq_step, [1,1,1,3])
            sbatch = np.tile(sbatch, [1,1,1,3])
            occ_map_step = np.concatenate([1-occ_map_step, np.zeros([(K+T)*seq_steps, image_size, image_size, 2])], axis=3)
            
            if include_road:
                maps_road[:,:,:,0] = np.zeros([maps_road.shape[0],maps_road.shape[1],maps_road.shape[2]]) #turning street cyan
                maps_road = maps_road*0.7 #reduce brightness
                maps_lines = maps_lines / 2.0 #turning lines gray

                maps_road_with_lines = np.copy(maps_road)
                for k in range(maps_road.shape[0]-1):
                    for l in range(maps_road.shape[1]):
                        for m in range(maps_road.shape[2]):
                            if maps_lines[k,l,m,0] > 0:
                                maps_road_with_lines[k,l,m] = maps_lines[k,l,m]
                            if occ_map_step[k,l,m,0] > 0:
                                maps_road_with_lines[k,l,m] = occ_map_step[k,l,m] * 0.2
                            if np.sum(sbatch[k,l,m,:]) <= 2:
                                sbatch[k,l,m] = maps_road_with_lines[k,l,m]
                            if np.sum(samples_seq_step[k,l,m,:]) <= 2:
                                samples_seq_step[k,l,m] = maps_road_with_lines[k,l,m]

                                
            else:
                samples_seq_step = np.maximum(occ_map_step * 0.2, samples_seq_step)
                sbatch = np.maximum(occ_map_step * 0.2, sbatch)
           
            pred_list = np.split(sbatch[:K, :, :, :], K, axis=0)
            pred_list = [np.squeeze(pred) for pred in pred_list]
            true_list = np.split(sbatch[:K+T, :, :, :], K+T, axis=0)
            true_list = [np.squeeze(true) for true in true_list]
            for t in range(K + T):
                
                pred = np.squeeze(samples_seq_step[t])
                target = np.squeeze(sbatch[t])
                pred = (pred * 255).astype("uint8")
                target = (target * 255).astype("uint8")

                cpsnr[t] = measure.compare_psnr(pred, target)

                pred = draw_frame(pred, t < K)
                #target = draw_frame(target, t < K)

                pred_list.append(pred)
                #true_list.append(target)
                tmp = (true_list[t]*255).astype("uint8")
                tmp = draw_frame(tmp, t < K)

                cv2.imwrite(savedir + "/pred_" +
                            "{0:04d}".format(t) + ".png", pred)
                cv2.imwrite(savedir + "/gt_" +
                            "{0:04d}".format(t) + ".png", tmp) #used to be target

            psnr_err = np.concatenate((psnr_err, cpsnr[None, K:]), axis=0)
            ssim_err = np.concatenate((ssim_err, cssim[None, K:]), axis=0)

            # Create GIFs of predicted sequence
            kwargs = {'duration': 0.2}
            imageio.mimsave(savedir + "/pred_" + str(i).zfill(3) +
                            ".gif", pred_list, 'GIF', **kwargs)
            imageio.mimsave(savedir + "/gt_" + str(i).zfill(3) +
                            ".gif", true_list, 'GIF', **kwargs)

        np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
        print("Results saved to " + save_path)
    print("Done.")
    
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
    parser.add_argument("--tfrecord", type=str, dest="tfrecord", default="all_in_one_imgsze=96_seqlen=4_K=9_T=10_all",
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

    args = parser.parse_args()
    main(**vars(args))
