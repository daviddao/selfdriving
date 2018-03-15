"""
Based on train_gridmap.py but with occlusion map as additional input of motion and content decoder
"""
import sys
import time
import imageio

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio

from move_network_distributed import MCNET
from utils import *
from load_gridmap import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed
import time
import _thread

next_seq_batch = None
next_input_batch = None
next_map_batch = None
next_transformation_batch = None
first_batch = True

def main(lr_D, lr_G, batch_size, alpha, beta, image_size, K,
         T, num_iter, gpu, sequence_steps, d_input_frames, useSELU=True,
         useCombinedMask=False, predOcclValue=1, img_save_freq=200):
    # "/lhome/phlippe/dataset/TwoHourSequence_crop/Train/compressed64x64/"
    data_path = '/mnt/ds3lab/daod/mercedes_benz/phlippe/dataset/BigLoop/'
    if sequence_steps * (K + T) <= 60:
      train_list = data_path + "train_onmove_96x96.txt"
    else:
      train_list = data_path + "train_onmove_long_96x96.txt"
    f = open(train_list, "r")
    trainfiles = f.readlines()
    print(str(len(trainfiles)) + " train files")
    margin = 0.3
    updateD = True
    updateG = True
    iters = 0
    iters_G = 0

    prefix = ("GRIDMAP_MCNET_onmove"
              + "_image_size=" + str(image_size)
              + "_K=" + str(K)
              + "_T=" + str(T)
              + "_seqsteps=" + str(sequence_steps)
              + "_batch_size=" + str(batch_size)
              + "_alpha=" + str(alpha)
              + "_beta=" + str(beta)
              + "_lr_G=" + str(lr_G)
              + "_lr_D=" + str(lr_D)
              + "_d_in=" + str(d_input_frames)
              + "_selu=" + str(useSELU)
              + "_comb=" + str(useCombinedMask)
              + "_predV=" + str(predOcclValue))

    print("\n" + prefix + "\n")
    checkpoint_dir = "../models/" + prefix + "/"
    samples_dir = "../samples/" + prefix + "/"
    summary_dir = "../logs/" + prefix + "/"

    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)
    if not exists(samples_dir):
        makedirs(samples_dir)
    if not exists(summary_dir):
        makedirs(summary_dir)

    print("Setup model...")
    model = MCNET(image_size=[image_size, image_size], c_dim=1,
                  K=K, batch_size=batch_size, T=T,
                  checkpoint_dir=checkpoint_dir,
                  iterations=sequence_steps,
                  d_input_frames=d_input_frames,
                  useSELU=True, motion_map_dims=2,
                  showFutureMaps=True,
                  predOcclValue=predOcclValue,
                  gpu=gpu, useGAN=(beta != 0))

    print("Setup optimizer...")
    with tf.device("/gpu:%d" % gpu[0]):
        if beta != 0:
          d_optim = tf.train.AdamOptimizer(lr_D, beta1=0.5).minimize(
              model.d_loss, var_list=model.d_vars
          )
        else:
          d_optim = tf.constant(0.0)
        g_optim = tf.train.AdamOptimizer(lr_G, beta1=0.5).minimize(
            alpha * model.L_img + beta * model.L_GAN, var_list=model.g_vars
        )

    print("Setup session...")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                          log_device_placement=False,
                                          gpu_options=gpu_options)) as sess:

        tf.global_variables_initializer().run()

        print("Load model...")
        if model.load(sess, checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        g_sum = tf.summary.merge([model.L_p_sum,
                                  model.L_BCE_sum, model.loss_sum,
                                  model.L_GAN_sum])
        d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                                  model.d_loss_fake_sum])
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        counter = iters + 1
        start_time = time.time()



        with Parallel(n_jobs=batch_size) as parallel:
            def load_next_batch(batchidx):
                global next_seq_batch
                global next_input_batch
                global next_map_batch
                global next_transformation_batch

                seq_batch = np.zeros((batch_size, image_size, image_size,
                                      sequence_steps * (K + T), 2), dtype="float32")
                input_batch = np.zeros((batch_size, image_size, image_size,
                                      sequence_steps * (K + T), 2), dtype="float32")
                map_batch = np.zeros((batch_size, image_size, image_size,
                                       sequence_steps * (K + T) + 1, 2), dtype="float32")
                transformation_batch = np.zeros((batch_size,
                                       sequence_steps * (K + T), 3, 8), dtype="float32")
                t0 = time.time()
                seq_steps = np.repeat(
                    np.array([1 + sequence_steps * (K + T)]), batch_size, axis=0)
                tfiles = np.array(trainfiles)[batchidx]
                shapes = np.repeat(
                    np.array([image_size]), batch_size, axis=0)
                combLoss = np.repeat(useCombinedMask, batch_size, axis=0)
                output = parallel(delayed(load_gridmap_onmove)(f, img_sze, seq, useCM)
                                  for f, img_sze, seq, useCM in zip(tfiles,
                                                             shapes,
                                                             seq_steps,
                                                             combLoss))
                for i in range(batch_size):
                    seq_batch[i] = output[i][0]
                    input_batch[i] = output[i][1]
                    map_batch[i] = output[i][2]
                    transformation_batch[i] = output[i][3]

                next_seq_batch = seq_batch
                next_input_batch = input_batch
                next_map_batch = map_batch
                next_transformation_batch = transformation_batch

            def fetch_next_batch(batchidx):
                global next_seq_batch
                global next_input_batch
                global next_map_batch
                global next_transformation_batch
                global first_batch

                wait_start_time = time.time()
                last_time = time.time()
                while not first_batch and (next_seq_batch is None or next_input_batch is None or next_map_batch is None or next_transformation_batch is None):
                    if (time.time() - last_time) > 0.1:
                        print("Waiting on next_grid/next_sequence_gt/next_sequence_grid, time=" + str(time.time() - wait_start_time))
                        last_time = time.time()
                seq_batch = (next_seq_batch)
                input_batch = (next_input_batch)
                map_batch = (next_map_batch)
                transformation_batch = (next_transformation_batch)

                next_seq_batch = None
                next_input_batch = None
                next_map_batch = None
                next_transformation_batch = None
                first_batch = False

                _thread.start_new_thread(load_next_batch, (batchidx,))
                return (seq_batch, input_batch, map_batch, transformation_batch)

            while iters < num_iter:
                mini_batches = get_minibatches_idx(
                    len(trainfiles), batch_size, shuffle=True)
                for _, batchidx in mini_batches:
                    if len(batchidx) == batch_size:
                        load_start = time.time()
                        if first_batch:
                            fetch_next_batch(batchidx)
                        seq_batch, input_batch, map_batch, transformation_batch = fetch_next_batch(batchidx)
                        print("Loading done in "+str(time.time() - load_start)+"sec.")

                        if beta != 0 and updateD:
                            _, summary_str = sess.run([d_optim, d_sum],
                                                      feed_dict={model.input_tensor: input_batch,
                                                                 model.motion_map_tensor: map_batch,
                                                                 model.target: seq_batch,
                                                                 model.ego_motion: transformation_batch})
                            writer.add_summary(summary_str, counter)

                        if updateG:
                            _, summary_str, max_pred, min_pred, max_labels, min_labels = sess.run([g_optim, g_sum, model.max_pred, model.min_pred, model.max_labels, model.min_labels],
                                                      feed_dict={model.input_tensor: input_batch,
                                                                 model.motion_map_tensor: map_batch,
                                                                 model.target: seq_batch,
                                                                 model.ego_motion: transformation_batch})
                            writer.add_summary(summary_str, counter)
                            iters_G += 1
                        iters += 1

                        # Usage of WGAN-GP
                        errD = model.d_loss.eval({model.input_tensor: input_batch,
                                                  model.motion_map_tensor: map_batch,
                                                  model.target: seq_batch,
                                                  model.ego_motion: transformation_batch})
                        errD_fake = model.d_loss_fake.eval({model.input_tensor: input_batch,
                                                            model.motion_map_tensor: map_batch,
                                                            model.target: seq_batch,
                                                            model.ego_motion: transformation_batch})
                        errD_real = model.d_loss_real.eval({model.input_tensor: input_batch,
                                                            model.motion_map_tensor: map_batch,
                                                            model.target: seq_batch,
                                                            model.ego_motion: transformation_batch})
                        errG = model.L_GAN.eval({model.input_tensor: input_batch,
                                                 model.motion_map_tensor: map_batch,
                                                 model.target: seq_batch,
                                                 model.ego_motion: transformation_batch})
                        img_err = model.L_img.eval({model.input_tensor: input_batch,
                                                    model.motion_map_tensor: map_batch,
                                                    model.target: seq_batch,
                                                    model.ego_motion: transformation_batch})

                        training_models_info = ""
                        if not updateD:
                            training_models_info += ", D not trained"
                        if not updateG:
                            training_models_info += ", G not trained"

                        if errD_fake < margin or errD_real < margin:
                          updateD = False
                        if errD_fake > (1.-margin) or errD_real > (1.-margin):
                          updateG = False
                        if True:#not updateD and not updateG:
                          updateD = True
                          updateG = True

                        counter += 1


                        print(
                            "Iters: [%5d|%5d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f, img_loss: %.8f %s"
                            % (iters_G, iters, time.time() - start_time, errD, errG, img_err, training_models_info)
                        )
                        print(
                            "Iters: [%5d|%5d] d_fake: %.8f, d_real: %.8f, maxPred: %4.4f, minPred: %4.4f, maxLabels: %4.4f, minLabels: %4.4f"
                            % (iters_G, iters, errD_fake, errD_real, max_pred, min_pred, max_labels, min_labels)
                        )

                        if np.mod(counter, img_save_freq) == 1:
                            print(np.reshape(transformation_batch[:,:,0,:6], [transformation_batch.shape[0], transformation_batch.shape[1], 2, 3]))
                            samples, samples_trans, samples_pre_trans, target_occ, motion_maps, occ_map = sess.run([model.G, model.G_trans, model.G_before_trans, model.target, model.motion_map_tensor, model.loss_occlusion_mask],
                                                                    feed_dict={model.input_tensor: input_batch,
                                                                               model.motion_map_tensor: map_batch,
                                                                               model.target: seq_batch,
                                                                               model.ego_motion: transformation_batch})

                            for seq_step in range(sequence_steps * 2):
                                start_frame = seq_step // 2 * (K + T)
                                end_frame = start_frame + K
                                if seq_step % 2 == 1:
                                  start_frame += K
                                  end_frame += T
                                frame_count = end_frame - start_frame

                                maps_lines = (motion_maps[0, :, :, start_frame:start_frame+1, 0:1].swapaxes(0, 2).swapaxes(1, 2) + 1) // 2
                                maps_road = (motion_maps[0, :, :, start_frame:start_frame+1, 1:2].swapaxes(0, 2).swapaxes(1, 2) + 1) // 2
                                samples_seq_step = (samples[
                                    0, :, :,start_frame: end_frame].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
                                samples_trans_seq_step = (samples_trans[
                                    0, :, :,start_frame: end_frame].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
                                samples_pre_trans_seq_step = (samples_pre_trans[
                                    0, :, :,start_frame: end_frame].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
                                sbatch = (target_occ[
                                    0, :, :,start_frame: end_frame, 0:1].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
                                occ_map_step = occ_map[
                                    0, :, :,start_frame: end_frame].swapaxes(0, 2).swapaxes(1, 2)

                                maps_lines = np.tile(maps_lines, [1,1,1,3])
                                maps_road = np.tile(maps_road, [1,1,1,3])
                                samples_seq_step = np.tile(samples_seq_step, [1,1,1,3])
                                samples_trans_seq_step = np.tile(samples_trans_seq_step, [1,1,1,3])
                                samples_pre_trans_seq_step = np.tile(samples_pre_trans_seq_step, [1,1,1,3])
                                sbatch = np.tile(sbatch, [1,1,1,3])

                                occ_map_step = np.concatenate([1-occ_map_step, np.zeros([frame_count, image_size, image_size, 2])], axis=3)
                                samples_seq_step = np.maximum(occ_map_step * 0.2, samples_seq_step)
                                samples_trans_seq_step = np.maximum(occ_map_step * 0.2, samples_trans_seq_step)
                                # print samples_trans_seq_step
                                sbatch = np.maximum(occ_map_step * 0.2, sbatch)

                                samples_seq_step = np.concatenate(
                                    (maps_lines, samples_seq_step, maps_road, sbatch, maps_lines, samples_trans_seq_step, maps_road, samples_pre_trans_seq_step), axis=0)
                                print("Saving sample ...")
                                save_images(samples_seq_step[:, :, :, :], [4, frame_count + 1],
                                            samples_dir + "train_" + str(iters).zfill(7) + "_" + str(seq_step) + ".png")
                        if np.mod(counter, 500) == 2:
                            print("#"*50)
                            print("Save model...")
                            print("#"*50)
                            model.save(sess, checkpoint_dir, counter)



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--lr_D", type=float, dest="lr_D",
                        default=0.0001, help="Base Learning Rate for Discriminator")
    parser.add_argument("--lr_G", type=float, dest="lr_G",
                        default=0.0001, help="Base Learning Rate for Generator")
    parser.add_argument("--batch_size", type=int, dest="batch_size",
                        default=4, help="Mini-batch size")
    parser.add_argument("--alpha", type=float, dest="alpha",
                        default=1.0, help="Image loss weight")
    parser.add_argument("--beta", type=float, dest="beta",
                        default=0.02, help="GAN loss weight")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=96, help="Training image size")
    parser.add_argument("--K", type=int, dest="K",
                        default=10, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=100000, help="Number of iterations")
    parser.add_argument("--seq_steps", type=int, dest="sequence_steps",
                        default=5, help="Number of iterations per Sequence (K | T | K | T | ...) - one K + T step is one iteration")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                        help="GPU device id")
    parser.add_argument("--d_input_frames", type=int, dest="d_input_frames",
                        default=20, help="How many frames the discriminator should get. Has to be at least K+T")
    parser.add_argument("--selu", type=str2bool, dest="useSELU",
                        default=True, help="If SELU should be used instead of RELU")
    parser.add_argument("--combMask", type=str2bool, dest="useCombinedMask",
                        default=False, help="If SELU should be used instead of RELU")
    parser.add_argument("--predOcclValue", type=int, dest="predOcclValue",
                        default=1, help="If SELU should be used instead of RELU")
    parser.add_argument("--imgFreq", type=int, dest="img_save_freq",
                        default=200, help="If SELU should be used instead of RELU")


    args = parser.parse_args()
    main(**vars(args))
