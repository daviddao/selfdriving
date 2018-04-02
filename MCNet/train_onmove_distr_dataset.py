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

from move_network_distributed_noGPU import MCNET
from utils import *
from load_gridmap import *
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
from joblib import Parallel, delayed
import time
import _thread
import threading
import datetime

#from tensorflow.python import debug as tf_debug

next_seq_batch = None
next_input_batch = None
next_map_batch = None
next_transformation_batch = None
first_batch = True

def main(lr_D, lr_G, batch_size, alpha, beta, image_size, K,
         T, num_iter, gpu, sequence_steps, d_input_frames, tfrecordname, useSELU=True,
         useCombinedMask=False, predOcclValue=1, img_save_freq=200, model_name="", useSharpen=False):
    # "/lhome/phlippe/dataset/TwoHourSequence_crop/Train/compressed64x64/"
    data_path = '/mnt/ds3lab/daod/mercedes_benz/phlippe/dataset/BigLoop/'
    data_path_scratch = '/mnt/ds3lab-scratch/lucala/phlippe/dataset/'
    model_path_scratch = '/mnt/ds3lab-scratch/lucala/phlippe/'
    """
    if sequence_steps * (K + T) <= 60:
      train_list = data_path + "train_onmove_96x96.txt"
    else:
      train_list = data_path + "train_onmove_long_96x96.txt"
    f = open(train_list, "r")
    trainfiles = f.readlines()
    print(str(len(trainfiles)) + " train files")
    """
    margin = 0.3
    updateD = True
    updateG = True
    iters = 0
    iters_G = 0

    date_now = datetime.datetime.now()

    prefix = (model_name
            + "GRIDMAP_MCNET_onmove"
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
              + "_predV=" + str(predOcclValue)
              + "_datetime=" + str(date_now.hour)+":"+str(date_now.minute)+"-"+str(date_now.day)+"-"+str(date_now.month)+"-"+str(date_now.year))

    print("\n" + prefix + "\n")
    checkpoint_dir = model_path_scratch + "models/" + prefix + "/"
    samples_dir = model_path_scratch + "samples/" + prefix + "/"
    summary_dir = model_path_scratch + "logs/" + prefix + "/"

    if not exists(checkpoint_dir):
        makedirs(checkpoint_dir)
    if not exists(samples_dir):
        makedirs(samples_dir)
    if not exists(summary_dir):
        makedirs(summary_dir)
        
    num_gpu = len(gpu)
    
    
    def average_gradients(tower_grads): #taken from cifar10_multi_gpu example
        """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
        tower_grads: List of lists of (gradient, variable) tuples. The outer list
          is over individual gradients. The inner list is over the gradient
          calculation for each tower.
        Returns:
         List of pairs of (gradient, variable) where the gradient has been averaged
         across all towers.


        """
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            # Note that each grad_and_vars looks like the following:
            #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
            grads = []
            for g, _ in grad_and_vars:
                # Add 0 dimension to the gradients to represent the tower.
                expanded_g = tf.expand_dims(g, 0)

                # Append on a 'tower' dimension which we will average over below.
                grads.append(expanded_g)

            # Average over the 'tower' dimension.
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    
    
    graph = tf.Graph()
    
    with graph.as_default():  
        
        print("Setup dataset...")
        #def input_fn():
        def _parse_function(example_proto):
            keys_to_features = {#'input_batch_shape': tf.FixedLenFeature((), tf.int64),
                                #'seq_batch_shape': tf.FixedLenFeature((), tf.int64),
                                #'maps_batch_shape': tf.FixedLenFeature((), tf.int64),
                                #'transformation_batch_shape': tf.FixedLenFeature((), tf.int64),
                                'input_seq': tf.FixedLenFeature((), tf.string),
                                'target_seq': tf.FixedLenFeature((), tf.string),
                                'maps': tf.FixedLenFeature((), tf.string),
                                'tf_matrix': tf.FixedLenFeature((), tf.string)}

            parsed_features = tf.parse_single_example(example_proto, keys_to_features)
            #input_batch_shape = parsed_features['input_batch_shape']
            #seq_batch_shape = parsed_features['seq_batch_shape']
            #maps_batch_shape = parsed_features['maps_batch_shape']
            #transformation_batch_shape = parsed_features['transformation_batch_shape']
            input_batch_shape = [96, 96, 76, 2]
            seq_batch_shape = [96, 96, 76, 2]
            maps_batch_shape = [96, 96, 77, 2]
            transformation_batch_shape = [76,3,8]
            input_seq = tf.reshape(tf.decode_raw(parsed_features['input_seq'], tf.float32), input_batch_shape, name='reshape_input_seq')
            target_seq = tf.reshape(tf.decode_raw(parsed_features['target_seq'], tf.float32), seq_batch_shape, name='reshape_target_seq')
            maps = tf.reshape(tf.decode_raw(parsed_features['maps'], tf.float32), maps_batch_shape, name='reshape_maps')
            tf_matrix = tf.reshape(tf.decode_raw(parsed_features['tf_matrix'], tf.float32), transformation_batch_shape, name='reshape_tf_matrix')

            return input_seq, target_seq, maps, tf_matrix

        dataset = tf.data.TFRecordDataset([data_path_scratch + tfrecordname + '.tfrecord'])
        #tfrecordsLoc = data_path_scratch + "tfrecords/"
        #files = tf.data.Dataset.list_files(tfrecordsLoc + "*.tfrecord").shuffle(len(trainfiles))
        #dataset = files.interleave(lambda x: tf.data.TFRecordDataset(x).prefetch(100),cycle_length=200)
        #map trainfiles to numbers by extracting last part
        dataset = dataset.map(_parse_function, num_parallel_calls=64)
        #shuffle and enhance dataset by num_iter amount
        dataset = dataset.shuffle(1000).repeat(num_iter) #initially 10k but too much memory, process is killed
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(num_gpu*2)
            #iterator = dataset.make_one_shot_iterator()
            #seq_batch, input_batch, map_batch, transformation_batch = iterator.get_next()
            #return seq_batch, input_batch, map_batch, transformation_batch

        print("Setup optimizer...")
        #g_optim = tf.train.AdamOptimizer(lr_G, beta1=0.5).minimize(alpha * model.L_img, var_list=model.g_vars)
        if beta != 0:
            opt_D = tf.train.AdamOptimizer(lr_D, beta1=0.5)

        opt_E = tf.train.AdamOptimizer(lr_G, beta1=0.5)
        
        
        # Create a variable to count number of train calls
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        
        # These are the lists of gradients for each tower 
        tower_grads = []
        if beta != 0:
            tower_grads_d = []

        print("Setup model...")
        model = MCNET(image_size=[image_size, image_size], c_dim=1,
                      K=K, batch_size=batch_size, T=T,
                      checkpoint_dir=checkpoint_dir,
                      iterations=sequence_steps,
                      d_input_frames=d_input_frames,
                      useSELU=True, motion_map_dims=2,
                      showFutureMaps=True,
                      predOcclValue=predOcclValue,
                      gpu=-1, useGAN=(beta != 0), useSharpen=useSharpen) #gpu dummy value

        
        input_shape = [batch_size*num_gpu, model.image_size[0],model.image_size[1], sequence_steps * (K + T), model.c_dim + 1]
        motion_map_shape = [batch_size*num_gpu, model.image_size[0],model.image_size[1], sequence_steps * (K + T) + model.maps_offset, 2] #last arg. motion_map_dims=2
        target_shape = [batch_size*num_gpu, model.image_size[0],model.image_size[1], sequence_steps * (K + T), model.c_dim + 1]  # +occlusion map for cutting out the gradients
        ego_motion_shape = [batch_size*num_gpu, sequence_steps * (K + T), 3, 8]

        model.pred_occlusion_map = tf.ones(model.occlusion_shape, dtype=tf.float32, name='Pred_Occlusion_Map') * model.predOcclValue
        
        #create iterator
        iterator = dataset.make_initializable_iterator()

        model.saver = tf.train.Saver(max_to_keep=10)
        print("Setup GPUs...")
        print("Using "+str(num_gpu)+" GPUs...")
        # Define the network for each GPU
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for i in range(len(gpu)):
                  with tf.device('/device:GPU:%d' % gpu[i]):
                        with tf.name_scope('Tower_%d' % (i)) as scope:
                            seq_batch, input_batch, map_batch, transformation_batch = iterator.get_next()
                            
                            # Construct the model
                            pred, trans_pred, pre_trans_pred = model.forward(input_batch, map_batch, transformation_batch,i)

                            # Calculate the loss for this tower   
                            #SSE_loss, KL_loss, D_loss, G_loss, LL_loss = loss(next_batch, x_tilde, z_x_log_sigma_sq, z_x_mean, d_x, d_x_p, l_x, l_x_tilde, dim1, dim2, dim3)
                            model.target = seq_batch
                            model.motion_map_tensor = map_batch
                            model.G = tf.stack(axis=3, values=pred)
                            model.G_trans = tf.stack(axis=3, values=trans_pred)
                            model.G_before_trans = tf.stack(axis=3, values=pre_trans_pred)
                            model.loss_occlusion_mask = (tf.tile(seq_batch[:, :, :, :, -1:], [1, 1, 1, 1, model.c_dim]) + 1) / 2.0
                            model.target_masked = model.mask_black(seq_batch[:, :, :, :, :model.c_dim], model.loss_occlusion_mask)
                            model.G_masked = model.mask_black(model.G, model.loss_occlusion_mask)
                            
                            if beta != 0:
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
                                                            
                            
                            # specify loss to parameters
                            model.t_vars = tf.trainable_variables()
                            model.g_vars = [var for var in model.t_vars if 'DIS' not in var.name]
                            if beta != 0:
                                model.d_vars = [var for var in model.t_vars if 'DIS' in var.name]

                            # Calculate the losses specific to encoder, generator, decoder
                            #L_e = tf.clip_by_value(KL_loss*KL_param + LL_loss, -100, 100)                            
                            model.L_img = model.weighted_BCE_loss(model.G_masked, model.target_masked) #cross-entropy mean
                            model.L_BCE = model.L_img
                            if (beta != 0): #use GAN
                                model.L_GAN = -tf.reduce_mean(model.D_)
                                model.d_loss = model.d_loss_fake + model.d_loss_real
                            else:
                                model.d_loss_fake = tf.constant(0.0)
                                model.d_loss_real = tf.constant(0.0)
                                model.d_loss = tf.constant(0.0)
                                model.L_GAN = tf.constant(0.0)
                                
                            model.L_p = tf.reduce_mean(tf.square(model.G_masked - model.target_masked))
                            
                            # Assemble all of the losses for the current tower only.
                            #losses = tf.get_collection('losses', scope)
                            # Calculate the total loss for the current tower.
                            #total_loss = tf.add_n(losses, name='total_loss')
                            
                            # Reuse variables for the next tower.
                            tf.get_variable_scope().reuse_variables()
                            
                            model.loss_sum = tf.summary.scalar("L_img", model.L_img)
                            model.L_p_sum = tf.summary.scalar("L_p", model.L_p)
                            model.L_BCE_sum = tf.summary.scalar("L_BCE", model.L_BCE)
                            model.L_GAN_sum = tf.summary.scalar("L_GAN", model.L_GAN)
                            model.d_loss_sum = tf.summary.scalar("d_loss", model.d_loss)
                            model.d_loss_real_sum = tf.summary.scalar("d_loss_real", model.d_loss_real)
                            model.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", model.d_loss_fake)
                            
                            summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                            # Calculate the gradients for the batch of data on this tower.
                            curr_grad = opt_E.compute_gradients(alpha * model.L_img + beta * model.L_GAN, var_list = model.g_vars)

                            # Keep track of the gradients across all towers.
                            tower_grads.append(curr_grad)
                            
                            if beta != 0:
                                curr_grad_d = opt_D.compute_gradients(model.d_loss, var_list=model.d_vars)
                                tower_grads_d.append(curr_grad_d)


    with graph.as_default():
        #with tf.device('/cpu:0'):
        # Average the gradients
        grads = average_gradients(tower_grads)
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        # apply the gradients with our optimizers
        train = opt_E.apply_gradients(grads, global_step=global_step)

        if beta != 0:
            grads_d = average_gradients(tower_grads_d)
            for grad, var in grads_d:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients_d', grad))
            train_d = opt_D.apply_gradients(grads_d, global_step=global_step)


        # Add histograms for trainable variables.
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        
        
    print("Setup session...")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)
    #with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
    #                                      log_device_placement=False,
    #                                      gpu_options=gpu_options)) as sess:
    with graph.as_default():
        init = tf.global_variables_initializer()
        #saver = tf.train.Saver() # initialize network saver
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options))

        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        
        sess.run(init)

        print("Load model...")
        if model.load(sess, checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        #g_sum = tf.summary.merge([model.L_p_sum,
        #                          model.L_BCE_sum, model.loss_sum,
        #                          model.L_GAN_sum])
        g_sum = tf.summary.merge(summaries)
        if beta != 0:
            d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                                      model.d_loss_fake_sum])
        writer = tf.summary.FileWriter(summary_dir, sess.graph)
        #print("Writer set...")
        counter = iters + 1
        start_time = time.time()

        #next_batch = input_fn()
        
        #initial assignment, dummy value
        max_pred = min_pred = max_labels = min_labels = img_err = -1
        errD = errD_fake = errD_real = errG = 0
        
        #num_iter_updated = num_iter // gpu
        while iters < num_iter:
            #print("initializing iterator...")
            sess.run(iterator.initializer)
            
            #for _, batchidx in mini_batches:
            for i in range(len(trainfiles)//(batch_size*num_gpu)):
                load_start = time.time()
                #seq_batch, input_batch, map_batch, transformation_batch = sess.run(next_batch)
                if beta != 0 and updateD:
                    _, summary_str = sess.run([train_d, d_sum])
                    writer.add_summary(summary_str, counter)

                if updateG:
                    _, summary_str, max_pred, min_pred, max_labels, min_labels = sess.run([train, g_sum, model.max_pred, model.min_pred, model.max_labels, model.min_labels])
                    writer.add_summary(summary_str, counter)
                    iters_G += 1
                iters += 1
                print("Run done in "+str(time.time() - load_start)+"sec.")
                
                errD = model.d_loss.eval(session=sess)
                errD_fake = model.d_loss_fake.eval(session=sess)
                errD_real = model.d_loss_real.eval(session=sess)
                errG = model.L_GAN.eval(session=sess)
                img_err = model.L_img.eval(session=sess)
                
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
                    #print(np.reshape(transformation_batch[:,:,0,:6], [transformation_batch.shape[0], transformation_batch.shape[1], 2, 3]))
                    samples, samples_trans, samples_pre_trans, target_occ, motion_maps, occ_map = sess.run([model.G, model.G_trans, model.G_before_trans, model.target, model.motion_map_tensor, model.loss_occlusion_mask])
                    for seq_step in range(sequence_steps * 2):
                        start_frame = seq_step // 2 * (K + T)
                        end_frame = start_frame + K
                        if seq_step % 2 == 1:
                          start_frame += K
                          end_frame += T
                        frame_count = end_frame - start_frame                            

                        maps_lines = (motion_maps[0, :, :, start_frame:start_frame+1, 0:1].swapaxes(0, 2).swapaxes(1, 2) + 1) // 2
                        maps_road = (motion_maps[0, :, :, start_frame:start_frame+1, 1:2].swapaxes(0, 2).swapaxes(1, 2) + 1) // 2
                        samples_seq_step = (samples[0, :, :,start_frame: end_frame].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
                        samples_trans_seq_step = (samples_trans[0, :, :,start_frame: end_frame].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
                        samples_pre_trans_seq_step = (samples_pre_trans[0, :, :,start_frame: end_frame].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
                        sbatch = (target_occ[0, :, :,start_frame: end_frame, 0:1].swapaxes(0, 2).swapaxes(1, 2) + 1) / 2.0
                        occ_map_step = occ_map[0, :, :,start_frame: end_frame].swapaxes(0, 2).swapaxes(1, 2)

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
                        default=1.001, help="Image loss weight")
    parser.add_argument("--beta", type=float, dest="beta",
                        default=0.0, help="GAN loss weight")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=96, help="Training image size")
    parser.add_argument("--K", type=int, dest="K",
                        default=9, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="Number of steps into the future")
    parser.add_argument("--num_iter", type=int, dest="num_iter",
                        default=100000, help="Number of iterations")
    parser.add_argument("--seq_steps", type=int, dest="sequence_steps",
                        default=4, help="Number of iterations per Sequence (K | T | K | T | ...) - one K + T step is one iteration")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                        help="GPU device id")
    parser.add_argument("--d_input_frames", type=int, dest="d_input_frames",
                        default=20, help="How many frames the discriminator should get. Has to be at least K+T")
    parser.add_argument("--selu", type=str2bool, dest="useSELU",
                        default=True, help="If SELU should be used instead of RELU")
    parser.add_argument("--combMask", type=str2bool, dest="useCombinedMask",
                        default=False, help="If SELU should be used instead of RELU")
    parser.add_argument("--predOcclValue", type=int, dest="predOcclValue",
                        default=-1, help="If SELU should be used instead of RELU")
    parser.add_argument("--imgFreq", type=int, dest="img_save_freq",
                        default=100, help="If SELU should be used instead of RELU")
    parser.add_argument("--prefix", type=str, dest="model_name",
                        default="", help="Prefix appended to model name for easier search")
    parser.add_argument("--sharpen", type=bool, dest="useSharpen",
                        default=False, help="If sharpening should be used")
    parser.add_argument("--tfrecord", type=str, dest="tfrecordname",
                        default="BigLoop2-5", help="tfrecord name")

    args = parser.parse_args()
    main(**vars(args))
