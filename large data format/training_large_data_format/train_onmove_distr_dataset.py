"""
Based on train_gridmap.py but with occlusion map as additional input of motion and content decoder
Code foundation was taken from cifar10 multi-gpu example.
"""
import sys
import time
import imageio
import glob

import tensorflow as tf
import scipy.misc as sm
import numpy as np
import scipy.io as sio

from move_network_distributed import MCNET
from utils import *
import os
from os import listdir, makedirs, system
from os.path import exists
from argparse import ArgumentParser
import time
import datetime
from PIL import Image

def main(lr_D, lr_G, batch_size, alpha, beta, image_size, data_w, data_h, K,
         T, num_iter, gpu, sequence_steps, d_input_frames, tfrecordname, useSELU=True,
         useCombinedMask=False, predOcclValue=1, img_save_freq=200, model_name="",
         useSharpen=False, useDenseBlock=True, samples=1, data_path_scratch="", model_path_scratch=""):

    # add / if string does not have
    data_path_scratch = data_path_scratch if data_path_scratch[-1] == "/" else data_path_scratch+"/"
    model_path_scratch = model_path_scratch if model_path_scratch[-1] == "/" else model_path_scratch+"/"

    # extract information from tfrecord string if none provided
    tfRecordFolder = False
    if tfrecordname == "":
        tfRecordFolder = True
        tfrecordname = glob.glob(data_path_scratch+'/*.tfrecord')[0].split('/')[-1].split('.')[0]
        print("Info: No TFRecord name provided. Using random TFRecord name in directory to parse info: " + tfrecordname)

    margin = 0.3
    updateD = True
    updateG = True
    iters = 0
    iters_G = 0

    date_now = datetime.datetime.now()

    # prefix string for model
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
              + "_predV=" + str(predOcclValue))

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

    # in case we are continuing training, extract previous value to continue counting
    try:
        tmp = sorted(glob.glob(samples_dir+'*.png'))[-1]
        tmp = tmp.split('/')[-1]
        tmp = tmp.split('_')[-2]
        prevNr = 0 if (tmp.lstrip('0') == '') else int(tmp.lstrip('0'))
    except:
        prevNr = 0

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
            for g, var in grad_and_vars:
                #fix for None-gradient
                g = g if g is not None else tf.zeros_like(var)
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
        # extract meta data from tfrecord name directly
        imgsze_tf, seqlen_tf, K_tf, T_tf, fc_tf, nr_samples, datasze_tf = parse_tfrecord_name(tfrecordname)
        assert(data_w == datasze_tf[0] and data_h == datasze_tf[1])
        assert(image_size == imgsze_tf)
        assert(sequence_steps <= seqlen_tf)
        assert(K <= K_tf)
        assert(T <= T_tf)

        # parser to read TFRecord file
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

            if (K+T)*sequence_steps < input_seq.shape[2]:
                target_seq = target_seq[:,:,:(K+T)*sequence_steps,:]
                input_seq = input_seq[:,:,:(K+T)*sequence_steps,:]
                maps = maps[:,:,:(K+T)*sequence_steps+1,:]
                tf_matrix = tf_matrix[:(K+T)*sequence_steps,:,:]

            rgb_cam = rgb_cam[:(K+T)*sequence_steps,:,:,:]
            seg_cam = seg_cam[:(K+T)*sequence_steps,:,:,:]
            dep_cam = tf.expand_dims(dep_cam[:(K+T)*sequence_steps,:,:],-1)
            direction = tf.cast(direction[:(K+T)*sequence_steps,:], tf.float32)

            speedyaw = tf.convert_to_tensor([inverseTransMat(tf_matrix[y,:]) for y in range((K+T)*sequence_steps)],dtype=tf.float32)

            return target_seq, input_seq, maps, tf_matrix, rgb_cam, seg_cam, dep_cam, direction, speedyaw

        # extract data from folder containing TFRecords or from single tfrecord
        if tfRecordFolder:
            num_records = len(os.listdir(data_path_scratch))
            print("Loading from directory. " + str(num_records) + " tfRecords found.")
            len_trainfiles = num_records * nr_samples
            files = tf.data.Dataset.list_files(data_path_scratch + "*.tfrecord").shuffle(num_records)
            dataset = files.apply(
                tf.contrib.data.parallel_interleave(lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=256, buffer_size=8*1024*1024),cycle_length=32, sloppy=True))
        else:
            print("Loading from single tfRecord. " + str(nr_samples) + " entries in tfRecord.")
            len_trainfiles = nr_samples
            dataset = tf.data.TFRecordDataset([data_path_scratch + tfrecordname + '.tfrecord'])

        # create the dataset
        dataset = dataset.map(_parse_function, num_parallel_calls=128)
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(1000))
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(num_gpu*128)

        print("Setup optimizer...")

        if beta != 0:
            opt_D = tf.train.AdamOptimizer(lr_D, beta1=0.5)

        opt_E = tf.train.AdamOptimizer(lr_G, beta1=0.5)
        opt_C = tf.train.AdamOptimizer(lr_G, beta1=0.5)

        # Create a variable to count number of train calls
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)

        # These are the lists of gradients for each tower
        tower_grads = []
        tower_grads_cam = []
        if beta != 0:
            tower_grads_d = []

        print("Setup model...")
        model = MCNET(image_size=[image_size, image_size],
                      data_size=[data_h, data_w], c_dim=1,
                      K=K, batch_size=batch_size, T=T,
                      checkpoint_dir=checkpoint_dir,
                      iterations=sequence_steps,
                      d_input_frames=d_input_frames,
                      useSELU=True, motion_map_dims=2,
                      showFutureMaps=True,
                      predOcclValue=predOcclValue,
                      useGAN=(beta != 0),
                      useSharpen=useSharpen,
                      useDenseBlock=useDenseBlock,
                      samples=samples)

        input_shape = [batch_size, model.image_size[0],model.image_size[1], sequence_steps * (K + T), model.c_dim + 1]
        motion_map_shape = [batch_size, model.image_size[0],model.image_size[1], sequence_steps * (K + T) + model.maps_offset, 2] #last arg. motion_map_dims=2
        target_shape = [batch_size, model.image_size[0],model.image_size[1], sequence_steps * (K + T), model.c_dim + 1]  # +occlusion map for cutting out the gradients
        ego_motion_shape = [batch_size, sequence_steps * (K + T), 3, 8]

        model.pred_occlusion_map = tf.ones(model.occlusion_shape, dtype=tf.float32, name='Pred_Occlusion_Map') * model.predOcclValue

        #create iterator
        iterator = dataset.make_initializable_iterator()

        loc = tf.zeros(model.gf_dim)
        scale = tf.ones(model.gf_dim)
        prior = tf.contrib.distributions.MultivariateNormalDiag(loc, scale)

        print("Setup GPUs...")
        print("Using "+str(num_gpu)+" GPUs...")
        # Define the network for each GPU
        with tf.variable_scope(tf.get_variable_scope()) as vscope:
            for i in range(len(gpu)):
                  with tf.device('/device:GPU:%d' % gpu[i]):
                        with tf.name_scope('Tower_%d' % (i)) as scope:
                            seq_batch, input_batch, map_batch, transformation_batch, rgb_cam, seg_cam, dep_cam, direction, speedyaw = iterator.get_next()

                            # Construct the model
                            pred, gm_pred, pre_gm_pred, rgb_pred, seg_pred, dep_pred, grid_posterior, img_posterior, trans_pred, dir_pred, speedyaw_pred = model.forward(input_batch, map_batch, transformation_batch, rgb_cam, seg_cam, dep_cam, direction, i)

                            # Calculate the loss for this tower
                            model.target = seq_batch
                            model.motion_map_tensor = map_batch
                            model.G = tf.stack(axis=3, values=pred)
                            model.G_trans = tf.stack(axis=3, values=gm_pred)
                            model.G_before_trans = tf.stack(axis=3, values=pre_gm_pred)
                            model.loss_occlusion_mask = (tf.tile(seq_batch[:, :, :, :, -1:], [1, 1, 1, 1, model.c_dim]) + 1) / 2.0
                            model.target_masked = model.mask_black(seq_batch[:, :, :, :, :model.c_dim], model.loss_occlusion_mask)
                            model.G_masked = model.mask_black(model.G, model.loss_occlusion_mask)
                            
                            model.dir_pred = tf.transpose(dir_pred,[1,0,2])
                            model.dir_gt = direction
                            
                            model.tmat_gt = transformation_batch
                            model.tmat_pred = tf.transpose(trans_pred,[1,0,2,3])
                            
                            model.speedyaw = speedyaw
                            model.speedyaw_pred = tf.transpose(speedyaw_pred,[1,0,2])
                            # using speedyaw error instead of transformation matrix error
                            #model.transformation_error = tf.reduce_mean(tf.squared_difference(transformation_batch, tf.transpose(trans_pred,[1,0,2,3])))
                            model.speedyaw_error = tf.reduce_mean(tf.squared_difference(model.speedyaw,model.speedyaw_pred))
                            model.direction_error = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.transpose(dir_pred,[1,0,2]),labels=direction))
                            model.odometry_error = model.direction_error + model.speedyaw_error
                            
                            model.rgb = rgb_cam
                            model.seg = seg_cam
                            model.dep = dep_cam
                            model.rgb_pred = tf.stack(rgb_pred,1)
                            model.seg_pred = tf.stack(seg_pred,1)
                            model.dep_pred = tf.stack(dep_pred,1)
                            if model.useDense:
                                model.rgb_diff = tf.reduce_mean(tf.squared_difference(model.rgb_pred, model.rgb))
                                model.seg_diff = tf.reduce_mean(tf.squared_difference(model.seg_pred, model.seg))
                                model.dep_diff = tf.reduce_mean(tf.squared_difference(model.dep_pred, model.dep))

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
                            if model.useDense:
                                model.L_img = -model.weighted_BCE_loss(model.G_masked, model.target_masked)
                            else:
                                divergence = tf.stack([tf.contrib.distributions.kl_divergence(post, prior) for post in grid_posterior],1)
                                model.L_img = model.weighted_BCE_loss(model.G_masked, model.target_masked)
                                model.L_img = -(tf.reduce_mean(model.L_img) - tf.reduce_mean(divergence)) #Loss = -ELBO = likelihood - divergence

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
                            if model.useDense: #using MSE
                                model.L_cam = model.rgb_diff + model.seg_diff + model.dep_diff
                                div_mean = tf.zeros([1])
                                L_cam_mean = model.L_cam
                            else:
                                rgb_loss = model.weighted_BCE_loss(model.rgb_pred,model.rgb)
                                seg_loss = model.weighted_BCE_loss(model.seg_pred,model.seg)
                                dep_loss = model.weighted_BCE_loss(model.dep_pred,model.dep)
                                model.L_cam = (rgb_loss + seg_loss + dep_loss)
                                divergence = tf.stack([tf.contrib.distributions.kl_divergence(post, prior) for post in img_posterior],1)
                                div_mean = tf.reduce_mean(divergence)
                                L_cam_mean = tf.reduce_mean(model.L_cam)
                                model.L_cam = -(L_cam_mean - div_mean) #Loss = -ELBO = likelihood - divergence
                            model.L_cam += model.odometry_error

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
                            curr_grad_cam = opt_C.compute_gradients(model.L_cam)#, var_list = model.c_vars)

                            # Keep track of the gradients across all towers.
                            tower_grads.append(curr_grad)
                            tower_grads_cam.append(curr_grad_cam)

                            if beta != 0:
                                curr_grad_d = opt_D.compute_gradients(model.d_loss, var_list=model.d_vars)
                                tower_grads_d.append(curr_grad_d)


    with graph.as_default():
        # Average the gradients
        grads = average_gradients(tower_grads)
        train = opt_E.apply_gradients(grads, global_step=global_step)

        if beta != 0:
            grads_d = average_gradients(tower_grads_d)
            train_d = opt_D.apply_gradients(grads_d, global_step=global_step)

        grads_c = average_gradients(tower_grads_cam)
        train_c = opt_C.apply_gradients(grads_c, global_step=global_step)

        model.saver = tf.train.Saver(max_to_keep=10)


    print("Setup session...")
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, allow_growth=True)

    with graph.as_default():
        init = tf.global_variables_initializer()

        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=False,gpu_options=gpu_options))

        sess.run(init)

        print("Load model...")
        if model.load(sess, checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        g_sum = tf.summary.merge(summaries)
        if beta != 0:
            d_sum = tf.summary.merge([model.d_loss_real_sum, model.d_loss_sum,
                                      model.d_loss_fake_sum])
        writer = tf.summary.FileWriter(summary_dir, sess.graph)

        counter = iters + 1
        start_time = time.time()

        #initial assignment, dummy value
        max_pred = min_pred = max_labels = min_labels = img_err = -1
        errD = errD_fake = errD_real = errG = 0

        # main training loop
        while iters < num_iter:
            sess.run(iterator.initializer)

            # loop over dataset
            for i in range(len_trainfiles//(batch_size*num_gpu)):

                load_start = time.time()
                if beta != 0 and updateD:
                    _, summary_str = sess.run([train_d, d_sum])
                    writer.add_summary(summary_str, counter)

                if updateG:
                    _, _, summary_str, max_pred, min_pred, max_labels, min_labels = sess.run([train, train_c, g_sum, model.max_pred, model.min_pred, model.max_labels, model.min_labels])
                    writer.add_summary(summary_str, counter)
                    iters_G += 1
                iters += 1
                print("Run done in "+str(time.time() - load_start)+"sec.")

                errD = model.d_loss.eval(session=sess)
                errD_fake = model.d_loss_fake.eval(session=sess)
                errD_real = model.d_loss_real.eval(session=sess)
                errG = model.L_GAN.eval(session=sess)
                img_err = model.L_img.eval(session=sess)
                cam_err = model.L_cam.eval(session=sess)
                #image VAE losses
                cam_div_mean_err = div_mean.eval(session=sess)
                cam_mean_err = L_cam_mean.eval(session=sess)
                odometry_error = model.odometry_error.eval(session=sess)
                speedyaw_error = model.speedyaw_error.eval(session=sess)

                # GAN training information
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

                # print current training information
                print(
                    "Iters: [%5d|%5d] time: %4.4f, d_loss: %.8f, L_GAN: %.8f, gridmap_loss: %.8f, images_loss: %.8f, elbo_recon_loss: %.8f, elbo_div_loss: %.8f, %s"
                    % (iters_G+prevNr, iters+prevNr, time.time() - start_time, errD, errG, img_err, cam_err, cam_mean_err, cam_div_mean_err, training_models_info)
                )
                print(
                    "Iters: [%5d|%5d] d_fake: %.8f, d_real: %.8f, maxPred: %4.4f, minPred: %4.4f, maxLabels: %4.4f, minLabels: %4.4f, odometry_loss: %.8f, speedyaw_error: %.8f"
                    % (iters_G+prevNr, iters+prevNr, errD_fake, errD_real, max_pred, min_pred, max_labels, min_labels, odometry_error, speedyaw_error)
                )

                # every img_save_freq save training sample and model
                if np.mod(counter, img_save_freq) == 1:
                    samples, samples_trans, samples_pre_trans, target_occ, motion_maps, occ_map, rgb, seg, dep, rgb_pred, seg_pred, dep_pred, sy, sy_pred, d_gt, d_pred, tmat_gt, tmat_pred = sess.run([model.G, model.G_trans, model.G_before_trans, model.target, model.motion_map_tensor, model.loss_occlusion_mask, model.rgb, model.seg, model.dep, model.rgb_pred, model.seg_pred, model.dep_pred, model.speedyaw, model.speedyaw_pred, model.dir_gt, model.dir_pred, model.tmat_gt, model.tmat_pred])

                    # dimensions are [batch_size, (K+T)*seq, 2]
                    txtsy = samples_dir + "speedyaw_" + str(counter+prevNr).zfill(7) + ".txt"
                    np.savetxt(txtsy, np.concatenate([sy[0], sy_pred[0]],axis=1))
                    
                    txttm = samples_dir + "transmat_" + str(counter+prevNr).zfill(7) + ".txt"
                    # transformation matrix dimensions are [batch_size, (K+T)*seq, 3, 8],
                    tmConcat = np.concatenate([np.reshape(tmat_gt[0],[tmat_gt.shape[1], 3*8]), np.reshape(tmat_pred[0],[tmat_gt.shape[1], 3*8])],axis=1)
                    # reshape makes [(K+T)*seq, 24*2] to [2*(K+T)*seq, 24], 1. line is gt, 2. prediction
                    np.savetxt(txttm, np.reshape(tmConcat,[2*tmat_gt.shape[1],3*8]))
                    
                    txtd = samples_dir + "direction_" + str(counter+prevNr).zfill(7) + ".txt"
                    # direction vector dimensions are [batch_size, (K+T)*seq, 2]
                    np.savetxt(txtd, np.concatenate([d_gt[0], d_pred[0]],axis=1))
                    
                    curr_frame = []
                    for seq_step in range(T):
                        curr_frame.append(np.concatenate([rgb[0,K+seq_step,:,:,:],rgb_pred[0,K+seq_step,:,:,:],
                                  seg[0,K+seq_step,:,:,:],seg_pred[0,K+seq_step,:,:,:],
                                  np.tile(dep[0,K+seq_step,:,:,:],[1,1,3]),np.tile(dep_pred[0,K+seq_step,:,:,:],[1,1,3])],1))
                    save_frame = Image.fromarray(np.uint8((np.concatenate(curr_frame,0)+1)/2*255))
                    save_frame.save(samples_dir + "train_cam_" + str(iters+prevNr).zfill(7)  + "_" + str(seq_step) + ".png")
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
                                    samples_dir + "train_" + str(iters+prevNr).zfill(7) + "_" + str(seq_step) + ".png")
                if np.mod(counter, img_save_freq) == 2:
                    print("#"*50)
                    print("Save model...")
                    print("#"*50)
                    model.save(sess, checkpoint_dir, counter+prevNr)


def tf_rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180

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
    parser.add_argument("--num-iter", type=int, dest="num_iter",
                        default=100000, help="Number of iterations")
    parser.add_argument("--seq-steps", type=int, dest="sequence_steps",
                        default=1, help="Number of iterations per Sequence (K | T | K | T | ...) - one K + T step is one iteration")
    parser.add_argument("--gpu", type=int, nargs="+", dest="gpu", required=True,
                        help="GPU device id")
    parser.add_argument("--d-input-frames", type=int, dest="d_input_frames",
                        default=20, help="How many frames the discriminator should get. Has to be at least K+T")
    parser.add_argument("--selu", type=str2bool, dest="useSELU",
                        default=True, help="If SELU should be used instead of RELU")
    parser.add_argument("--combMask", type=str2bool, dest="useCombinedMask",
                        default=False, help="If SELU should be used instead of RELU")
    parser.add_argument("--predOcclValue", type=int, dest="predOcclValue",
                        default=-1, help="If SELU should be used instead of RELU")
    parser.add_argument("--img-freq", type=int, dest="img_save_freq",
                        default=100, help="If SELU should be used instead of RELU")
    parser.add_argument("--prefix", type=str, dest="model_name",
                        default="", help="Prefix appended to model name for easier search")
    parser.add_argument("--sharpen", type=str2bool, dest="useSharpen",
                        default=False, help="If sharpening should be used. (deprecated)")
    parser.add_argument("--tfrecord", type=str, dest="tfrecordname",
                        default="", help="tfrecord name")
    parser.add_argument("--denseBlock", type=str2bool, dest="useDenseBlock", default=False,
                        help="Use DenseBlock (dil_conv) or VAE-distr.")
    parser.add_argument("--samples", type=int, dest="samples", default=1,
                        help="if using VAE how often should be sampled?")
    parser.add_argument("--data-path-scratch", type=str, dest="data_path_scratch", default="/mnt/ds3lab-scratch/lucala/dataset/CARLA/",
                        help="where are tfRecords stored?")
    parser.add_argument("--model-path-scratch", type=str, dest="model_path_scratch", default="/mnt/ds3lab-scratch/lucala/models/",
                        help="where are tfRecords stored?")
    parser.add_argument("--data-w", type=int, dest="data_w",
                        default=240, help="rgb/seg/depth image width size")
    parser.add_argument("--data-h", type=int, dest="data_h",
                        default=80, help="rgb/seg/depth image width size")

    args = parser.parse_args()
    main(**vars(args))
