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
import ssim
import imageio

import tensorflow as tf
import scipy.misc as sm
import scipy.io as sio
import numpy as np
import skimage.measure as measure

from velocity_net import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw


def main(data_path, prefix, image_size, K, T, gpu):
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
    f = open(data_path + "/test_data_list.txt", "r")
    testfiles = f.readlines()
    print("Start test with " + str(len(testfiles)) + " testfiles...")
    c_dim = 3  # Channel input for network

    checkpoint_dir = "../models/" + prefix + "/"
    best_model = None  # will pick last model

    # Setup model (for details see mcnet_deep_tracking.py)
    with tf.device("/gpu:%d" % gpu[0]):
        model = MCNET(image_size=[image_size, image_size], batch_size=1, K=K,
                      T=T, c_dim=1, checkpoint_dir=checkpoint_dir,
                      is_train=False, iterations=1, useSELU=True, motion_map_dims=2,
                      showFutureMaps=False)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options=gpu_options)) as sess:
        # Prepare model and directories
        tf.global_variables_initializer().run()

        quant_dir = "../results/quantitative/Gridmap/" + prefix + "/"
        save_path = quant_dir + "results_model=" + "best_model" + ".npz"
        if not exists(quant_dir):
            makedirs(quant_dir)

        if model.load(sess, checkpoint_dir, best_model):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed... exitting")
            return

        vid_names = []
        psnr_err = np.zeros((0, T))  # Array for saving the PSNR error of all test sequences
        ssim_err = np.zeros((0, T))  # Array for saving the SSIM error of all test sequences

        for i in range(len(testfiles)):
            print(testfiles[i])

            # Run prediction on single test sequence
            target_seq, input_seq, maps = load_gridmap_onmove(testfiles[i], image_size, K + T)
            target_batch = np.reshape(target_seq, (1, target_seq.shape[0], target_seq.shape[
                                   1], target_seq.shape[2], target_seq.shape[3]))
            input_batch = np.reshape(input_seq, (1, input_seq.shape[0], input_seq.shape[
                                    1], input_seq.shape[2], input_seq.shape[3]))
            maps_batch = np.reshape(maps, (1, maps.shape[0], maps.shape[
                                    1], maps.shape[2], maps.shape[3]))

            true_data = target_batch[:, :, :, :, :1].copy()
            pred_data = np.zeros(true_data.shape, dtype="float32")
            pred_data[0], occl_map = sess.run(model.G, model.loss_occlusion_mask
                                    feed_dict={model.input_tensor: input_batch,
                                               model.motion_map_tensor: maps_batch,
                                               model.target: target_batch})
            pred_image = np.zeros(true_data.shape, dtype="float32")
            pred_image = np.concatenate([pred_image]*3, axis=4)
            true_data = np.concatenate([true_data]*3, axis=4)
            pred_image = pred_image + (pred_image < 64) * np.concatenate([1-occl_map]*3, axis=4) * 64
            true_data = true_data + (true_data < 64) * np.concatenate([1-occl_map]*3, axis=4) * 64

            # Save predictions and PSNR/SSIM error
            savedir = "../results/images/Gridmap/" + prefix + "/" + \
                testfiles[i].split("/", -1)[-1].split(".", -1)[0]
            if not os.path.exists(savedir):
                os.makedirs(savedir)

            cpsnr = np.zeros((K + T,))
            cssim = np.zeros((K + T,))
            pred_list = np.split(target_batch[0, :, :, :K, 0:1], K, axis=2)
            pred_list = [np.squeeze(pred) for pred in pred_list]
            true_list = np.split(target_batch[0, :, :, :K, 0:1], K, axis=2)
            true_list = [np.squeeze(true) for true in true_list]
            for t in range(K + T):
                pred = (inverse_transform(
                    pred_data[0, :, :, t]) * 255).astype("uint8")
                target = (inverse_transform(
                    true_data[0, :, :, t]) * 255).astype("uint8")

                cpsnr[t] = measure.compare_psnr(pred, target)

                pred = draw_frame(pred, t < K)
                target = draw_frame(target, t < K)

                pred_list.append(pred)
                true_list.append(target)

                cv2.imwrite(savedir + "/pred_" +
                            "{0:04d}".format(t) + ".png", pred)
                cv2.imwrite(savedir + "/gt_" +
                            "{0:04d}".format(t) + ".png", target)

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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--prefix", type=str, dest="prefix", required=True,
                        help="Prefix for log/snapshot")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=128, help="Pre-trained model")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of input images")
    parser.add_argument("--T", type=int, dest="T",
                        default=40, help="Number of steps into the future")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                        help="GPU device id")
    parser.add_argument("--data_path", type=str, dest="data_path", default="/lhome/phlippe/dataset/",
                        help="Path where the test data is stored")

    args = parser.parse_args()
    main(**vars(args))
