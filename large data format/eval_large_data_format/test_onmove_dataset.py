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

from move_network_val_large_data_format import MCNET
from utils import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from skimage.draw import line_aa
from PIL import Image
from PIL import ImageDraw

def main(data_path, tfrecord, prefix, image_size, data_w, data_h, K, T, useGAN, useSharpen, num_gpu, include_road, num_iters,
seq_steps, useDenseBlock, samples, checkpoint_dir_loc, sy_loss, d_input_frames=20, predOcclValue=-1, beta=0, batch_size=1, cutout_input=0):

    gpu = np.arange(num_gpu)
    #need at least 1 gpu to run code
    assert(num_gpu>=1 and len(gpu)==num_gpu)

    print("Setup dataset...")
    imgsze_tf, seqlen_tf, K_tf, T_tf, fc_tf, nr_samples, datasze_tf = parse_tfrecord_name(tfrecord)
    assert(data_w == datasze_tf[0] and data_h == datasze_tf[1])
    assert(image_size == imgsze_tf)
    assert(seq_steps <= seqlen_tf)
    assert(K <= K_tf)
    assert(T <= T_tf)

    # parser for TFRecord files
    def _parse_function(example_proto):
        keys_to_features = {'input_seq': tf.FixedLenFeature((), tf.string),
                            'target_seq': tf.FixedLenFeature((), tf.string),
                            'maps': tf.FixedLenFeature((), tf.string),
                            'tf_matrix': tf.FixedLenFeature((), tf.string),
                            'rgb': tf.FixedLenFeature((), tf.string),
                            'segmentation': tf.FixedLenFeature((), tf.string),
                            'depth': tf.FixedLenFeature((), tf.string),
                            'direction': tf.FixedLenFeature((), tf.string)}

        parsed_features = tf.parse_single_example(example_proto, keys_to_features)

        input_batch_shape = [imgsze_tf, imgsze_tf, seqlen_tf*(K_tf+T_tf), 2]
        seq_batch_shape = [imgsze_tf, imgsze_tf, seqlen_tf*(K_tf+T_tf), 2]
        maps_batch_shape = [imgsze_tf, imgsze_tf, seqlen_tf*(K_tf+T_tf)+1, 2]
        transformation_batch_shape = [seqlen_tf*(K_tf+T_tf),3,8]
        rgb_batch_shape = [fc_tf,datasze_tf[1],datasze_tf[0],3]
        segmentation_batch_shape = [fc_tf,datasze_tf[1],datasze_tf[0],3]
        depth_batch_shape = [fc_tf,datasze_tf[1],datasze_tf[0]]
        direction_batch_shape = [fc_tf,2]

        input_seq = tf.reshape(tf.decode_raw(parsed_features['input_seq'], tf.float32), input_batch_shape, name='reshape_input_seq')
        target_seq = tf.reshape(tf.decode_raw(parsed_features['target_seq'], tf.float32), seq_batch_shape, name='reshape_target_seq')
        maps = tf.reshape(tf.decode_raw(parsed_features['maps'], tf.float32), maps_batch_shape, name='reshape_maps')
        tf_matrix = tf.reshape(tf.decode_raw(parsed_features['tf_matrix'], tf.float32), transformation_batch_shape, name='reshape_tf_matrix')

        rgb_cam = tf.reshape(tf.decode_raw(parsed_features['rgb'], tf.float32), rgb_batch_shape, name='reshape_rgb')
        seg_cam = tf.reshape(tf.decode_raw(parsed_features['segmentation'], tf.float32), segmentation_batch_shape, name='reshape_segmentation')
        dep_cam = tf.reshape(tf.decode_raw(parsed_features['depth'], tf.float32), depth_batch_shape, name='reshape_depth')

        direction = tf.reshape(tf.decode_raw(parsed_features['direction'], tf.uint8), direction_batch_shape, name='reshape_direction')

        # in case training parameters are set smaller than dataset parameters
        if (K+T)*seq_steps < input_seq.shape[2]:
            target_seq = target_seq[:,:,:(K+T)*seq_steps,:]
            input_seq = input_seq[:,:,:(K+T)*seq_steps,:]
            maps = maps[:,:,:(K+T)*seq_steps+1,:]
            tf_matrix = tf_matrix[:(K+T)*seq_steps,:,:]

        rgb_cam = rgb_cam[:(K+T)*seq_steps,:,:,:]
        seg_cam = seg_cam[:(K+T)*seq_steps,:,:,:]
        dep_cam = tf.expand_dims(dep_cam[:(K+T)*seq_steps,:,:],-1)
        direction = direction[:(K+T)*seq_steps,:]
        
        if cutout_input == 1:
            zero = tf.zeros(target_seq.shape)
            target_seq = tf.concat([zero[:,:,:K,:], target_seq[:,:,K:,:]], 2)
            input_seq = tf.concat([zero[:,:,:K,:], input_seq[:,:,K:,:]], 2)
            maps = tf.concat([zero[:,:,:K+1,:], maps[:,:,K+1:,:]], 2)
        elif cutout_input == 2:
            rgb_cam = tf.concat([tf.zeros(rgb_cam.shape)[:K], rgb_cam[K:]], 0)
        elif cutout_input == 3:
            seg_cam = tf.concat([tf.zeros(seg_cam.shape)[:K], seg_cam[K:]], 0)
        elif cutout_input == 4:
            dep_cam = tf.concat([tf.zeros(dep_cam.shape)[:K], dep_cam[K:]], 0)

        speedyaw = tf.convert_to_tensor([inverseTransMat(tf_matrix[y,:]) for y in range((K+T)*seq_steps)],dtype=tf.float32)

        return target_seq, input_seq, maps, tf_matrix, rgb_cam, seg_cam, dep_cam, direction, speedyaw

    tfrecordsLoc = data_path + tfrecord
    #loading from directory containing sharded tfrecords or from one tfrecord
    if os.path.isdir(tfrecordsLoc):
        num_records = len(os.listdir(tfrecordsLoc))
        print("Loading from directory. " + str(num_records) + " tfRecords found.")
        files = tf.data.Dataset.list_files(tfrecordsLoc + "/" + "*.tfrecord", shuffle=False)
        dataset = files.flat_map(lambda x: tf.data.TFRecordDataset(x))
    else:
        print("Loading from single tfRecord. " + str(nr_samples) + " entries in tfRecord.")
        dataset = tf.data.TFRecordDataset([tfrecordsLoc + '.tfrecord'])
    dataset = dataset.map(_parse_function)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))

    checkpoint_dir = checkpoint_dir_loc + prefix + "/"
    best_model = None  # will pick last model

    # initialize model
    model = MCNET(image_size=[image_size, image_size],
                  data_size=[data_h, data_w], c_dim=1,
                  K=K, batch_size=batch_size, T=T,
                  checkpoint_dir=checkpoint_dir,
                  iterations=seq_steps,
                  d_input_frames=d_input_frames,
                  useSELU=True, motion_map_dims=2,
                  showFutureMaps=True,
                  predOcclValue=predOcclValue,
                  useGAN=(beta != 0),
                  useSharpen=useSharpen,
                  useDenseBlock=useDenseBlock,
                  samples=samples, sy_loss=sy_loss)

    # Setup model and iterator
    model.pred_occlusion_map = tf.ones(model.occlusion_shape, dtype=tf.float32, name='Pred_Occlusion_Map') * model.predOcclValue
    iterator = dataset.make_initializable_iterator()
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        with tf.device("/gpu:%d" % gpu[0]):

            #fetch input
            seq_batch, input_batch, map_batch, transformation_batch, rgb_cam, seg_cam, dep_cam, direction, speedyaw = iterator.get_next()

            # Construct the model
            pred, trans_pred, pre_trans_pred, rgb_pred, seg_pred, dep_pred, trans_pred, dir_pred, speedyaw_pred = model.forward(input_batch, map_batch, transformation_batch, rgb_cam, seg_cam, dep_cam, direction)

            model.target = seq_batch
            model.motion_map_tensor = map_batch
            model.G = tf.stack(axis=3, values=pred)
            model.loss_occlusion_mask = (tf.tile(seq_batch[:, :, :, :, -1:], [1, 1, 1, 1, model.c_dim]) + 1) / 2.0
            model.target_masked = model.mask_black(seq_batch[:, :, :, :, :model.c_dim], model.loss_occlusion_mask)
            model.G_masked = model.mask_black(model.G, model.loss_occlusion_mask)

            model.sy = tf.reshape(speedyaw,[batch_size*(K+T)*seq_steps,2])
            model.sy_pred = tf.reshape(speedyaw_pred,[batch_size*(K+T)*seq_steps,2])
            model.rgb = rgb_cam
            model.seg = seg_cam
            model.dep = dep_cam
            model.rgb_pred = tf.stack(rgb_pred,1)
            model.seg_pred = tf.stack(seg_pred,1)
            model.dep_pred = tf.stack(dep_pred,1)

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


            if useGAN:
                model.L_GAN = -tf.reduce_mean(model.D_)
                model.d_loss = model.d_loss_fake + model.d_loss_real
            else:
                model.d_loss_fake = tf.constant(0.0)
                model.d_loss_real = tf.constant(0.0)
                model.d_loss = tf.constant(0.0)
                model.L_GAN = tf.constant(0.0)

            tf.get_variable_scope().reuse_variables()

    # add string to indicate whether horizon map is included in prediction
    if include_road:
        prefix = prefix + "_road_"
    else:
        prefix = prefix + "_noRoad_"

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options=gpu_options)) as sess:
        # Prepare model and directories
        tf.global_variables_initializer().run()

        model.saver = tf.train.Saver()

        # where to store results
        quant_dir = "../results/quantitative/Gridmap/" + prefix + "/"
        save_path = quant_dir + "results_model=" + "best_model" + ".npz"
        if not exists(quant_dir):
            makedirs(quant_dir)

        # load model
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
        for i in tqdm(range(num_iters)):

            samples, target_occ, motion_maps, occ_map, rgb, seg, dep, rgb_pred, seg_pred, dep_pred, sy, sy_pred = sess.run([model.G, model.target, model.motion_map_tensor, model.loss_occlusion_mask,
            model.rgb, model.seg, model.dep, model.rgb_pred, model.seg_pred, model.dep_pred, model.sy, model.sy_pred])

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

            # creating gt and prediction grid map gifs
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
                cssim[t] = measure.compare_ssim(pred, target, multichannel=True)

                pred = draw_frame(pred, t < K)

                pred_list.append(pred)
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

            # create compilation and gifs of image channels
            curr_frame = []
            curr_gif_pred = []
            curr_gif_gt = []
            rgb = (np.clip(rgb,-1,1)+1)/2 * 255
            seg = (seg+1)/2 * 255
            dep = (dep+1)/2 * 255
            rgb_pred = (np.clip(rgb_pred,-1,1)+1)/2 * 255
            seg_pred = (seg_pred+1)/2 * 255
            dep_pred = (dep_pred+1)/2 * 255
            for seq_step in range(K):
                curr_gif_pred.append(np.concatenate([rgb_pred[0,seq_step,:,:,:],seg_pred[0,seq_step,:,:,:],np.tile(dep_pred[0,seq_step,:,:,:],[1,1,3])],1))
                curr_gif_gt.append(np.concatenate([rgb[0,seq_step,:,:,:],seg[0,seq_step,:,:,:],np.tile(dep[0,seq_step,:,:,:],[1,1,3])],1))
            for seq_step in range(T):
                curr_frame.append(np.concatenate([np.asarray(Image.fromarray(np.uint8(true_list[K+seq_step]*255)).resize((data_h,data_h),Image.ANTIALIAS)),
                          np.asarray(Image.fromarray(pred_list[K*2+seq_step]).resize((data_h,data_h),Image.ANTIALIAS)),
                          rgb[0,K+seq_step,:,:,:],rgb_pred[0,K+seq_step,:,:,:],
                          seg[0,K+seq_step,:,:,:],seg_pred[0,K+seq_step,:,:,:],
                          np.tile(dep[0,K+seq_step,:,:,:],[1,1,3]),np.tile(dep_pred[0,K+seq_step,:,:,:],[1,1,3])],1))
                curr_gif_pred.append(np.concatenate([rgb_pred[0,K+seq_step,:,:,:],seg_pred[0,K+seq_step,:,:,:],np.tile(dep_pred[0,K+seq_step,:,:,:],[1,1,3])],1))
                curr_gif_gt.append(np.concatenate([rgb[0,K+seq_step,:,:,:],seg[0,K+seq_step,:,:,:],np.tile(dep[0,K+seq_step,:,:,:],[1,1,3])],1))
            save_frame = Image.fromarray(np.uint8(np.concatenate(curr_frame,0)))
            save_frame.save(savedir + "/img_" + str(i).zfill(3) + ".png")

            imageio.mimsave(savedir + "/img_pred_" + str(i).zfill(3) +
                            ".gif", curr_gif_pred, 'GIF', **kwargs)
            imageio.mimsave(savedir + "/img_gt_" + str(i).zfill(3) +
                            ".gif", curr_gif_gt, 'GIF', **kwargs)
            # save speed and yaw rate prediction
            if sy_loss:
                txtname = savedir + "/speedyaw_" + str(i).zfill(3) + ".txt"
                np.savetxt(txtname, np.concatenate([sy, sy_pred],axis=1))
        np.savez(save_path, psnr=psnr_err, ssim=ssim_err)
        print("Results saved to " + save_path)
    print("Done.")

def tf_rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180

#inverse transformation, see transformation in preprocessing script
def inverseTransMat(nextTf):
    z = tf.zeros([1,1], dtype=np.float32)
    matFull = tf.clip_by_value(nextTf,-1,1)
    #mean theta extracted from matrix, only use ARCSIN for +- range, take directly from [3,8]
    theta = -(tf.asin(-matFull[0,1])+tf.asin(matFull[0,3]))/2
    imsize = 96 // 2
    pixel_diff_y = matFull[1,2] * ((imsize - 1) / 2.0)
    pixel_diff_x = matFull[0,2] * ((imsize - 1) / 2.0)
    py = pixel_diff_y / tf.cos(theta)
    px = pixel_diff_x / tf.sin(theta)
    pixel_diff = tf.cond(tf.equal(tf.cos(theta),0), lambda: px, lambda: (px+py)/2)
    pixel_diff = tf.cond(tf.equal(tf.sin(theta),0), lambda: py, lambda: pixel_diff)
    pixel_size = 45.6 * 1.0 / imsize
    period_duration = 1.0 / 24
    vel = pixel_diff * pixel_size / period_duration
    yaw_rate = tf_rad2deg(theta) / period_duration
    return vel, yaw_rate

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
                        default="pure-sy_onlyCARLA_onmove_image_size=96_K=9_T=10_seqsteps=1_batch_size=4_alpha=1.001_beta=0.0_lr_G=0.0001_lr_D=0.0001_d_in=20_selu=True_comb=False_predV=-1", help="Prefix for log/snapshot")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=96, help="Pre-trained model")
    parser.add_argument("--K", type=int, dest="K",
                        default=9, help="Number of input images")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--useGAN", type=str2bool, dest="useGAN",
                        default=False, help="Model trained with GAN?")
    parser.add_argument("--useSharpen", type=str2bool, dest="useSharpen",
                        default=False, help="Model trained with sharpener? (deprecated)")
    parser.add_argument("--num-gpu", type=int, dest="num_gpu", default=1,
                        help="number of gpus (deprecated, does not use more than 1 GPU)")
    parser.add_argument("--data-path", type=str, dest="data_path", default="./tfrecords/",
                        help="Path where the test data is stored")
    parser.add_argument("--tfrecord", type=str, dest="tfrecord", default="evaldata_imgsze=96_fc=20_datasze=240x80_seqlen=1_K=9_T=10_size=20",
                        help="Either folder name containing tfrecords or name of single tfrecord.")
    parser.add_argument("--road", type=str2bool, dest="include_road", default=False,
                        help="Include horizon map. (DEPRECATED: no horizon maps in current dataset)")
    parser.add_argument("--num-iters", type=int, dest="num_iters", default=20,
                        help="How many files should be checked?")
    parser.add_argument("--seq-steps", type=int, dest="seq_steps", default=1,
                        help="Number of iterations in model.")
    parser.add_argument("--denseBlock", type=str2bool, dest="useDenseBlock", default=False,
                        help="Use DenseBlock (dil_conv) or VAE-distr.")
    parser.add_argument("--samples", type=int, dest="samples", default=1,
                        help="if using VAE how often should be sampled?")
    parser.add_argument("--chckpt-loc", type=str, dest="checkpoint_dir_loc", default="./model/",
                        help="Location of model checkpoint file")
    parser.add_argument("--data_w", type=int, dest="data_w",
                        default=240, help="rgb/seg/depth image width size")
    parser.add_argument("--data_h", type=int, dest="data_h",
                        default=80, help="rgb/seg/depth image width size")
    parser.add_argument("--speed-yaw-loss", type=str2bool, dest="sy_loss",
                        default=True, help="Add additional layer in network to compute speed yaw output?")
    parser.add_argument("--cutout-input", type=int, dest="cutout_input",
                        default=0, help="Remove channel from evaluation: 0 all input remains, 1 remove grid map, 2 remove RGB, 3 remove segmentation, 4 remove depth.")

    args = parser.parse_args()
    main(**vars(args))
