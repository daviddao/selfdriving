"""
Original Motion-Content Network code from https://github.com/rubenvillegas/iclr2017mcnet
- Adjusted for 3 channel difference and deep Tracking (occlusion map)
"""

import os
import tensorflow as tf

from BasicConvLSTMCell import BasicConvLSTMCell
from tensorflow.contrib.layers.python import layers as tf_layers
from ops import *
from additional_ops import *
from utils import *
from spatial_transformer import spatial_transformer_network


class MCNET(object):

    def __init__(self, image_size, data_size, batch_size=1, c_dim=1,  # Input has to be 2 dimensional: First occupancy, last occlusion map
                 K=10, T=10, checkpoint_dir=None,
                 iterations=1, d_input_frames=20, useSELU=False,
                 motion_map_dims=2, showFutureMaps=True,
                 predOcclValue=-1, useSmallSharpener=False,
                 useGAN=False, useSharpen=False, useDenseBlock=True, samples=1):

        self.samples = samples
        self.batch_size = batch_size
        self.image_size = image_size
        self.data_size = data_size
        self.useDenseBlock = useDenseBlock
        self.d_input_frames = d_input_frames
        self.useSELU = useSELU
        self.maps_offset = 1 if showFutureMaps else 0
        self.predOcclValue = predOcclValue
        self.small_sharpener = useSmallSharpener

        self.frame_rate = 24
        self.gridmap_size = 45.6

        self.useGAN = useGAN
        self.useSharpen = useSharpen

        self.gf_dim = 32
        self.df_dim = 64

        self.c_dim = c_dim
        self.K = K
        self.T = T
        self.iterations = iterations
        self.input_shape = [batch_size, self.image_size[0],
                             self.image_size[1], iterations * (K + T), c_dim + 1]
        self.motion_map_shape = [batch_size, self.image_size[0],
                             self.image_size[1], iterations * (K + T) + self.maps_offset, motion_map_dims]
        self.target_shape = [batch_size, self.image_size[0],
                             self.image_size[1], iterations * (K + T), c_dim + 1]  # In here also the occlusion map for cutting out the gradients
        self.occlusion_shape = [batch_size, self.image_size[0],
                                self.image_size[1], 1]
        self.ego_motion_shape = [batch_size, iterations * (K + T), 3, 8]
        self.rgb_shape = [batch_size, iterations * (K+T), self.data_size[0], self.data_size[1], 3]
        self.seg_shape = [batch_size, iterations * (K+T), self.data_size[0], self.data_size[1], 3]
        self.depth_shape = [batch_size, iterations * (K+T), self.data_size[0], self.data_size[1], 1]
        print("Target shape = " + str(self.target_shape))
        print("Ego motion shape = " + str(self.ego_motion_shape))
        print("RGB/Segmentation shape = " + str(self.rgb_shape))
        print("Depth shape = " + str(self.depth_shape))
        print("K = " + str(K) + ", T = " + str(T))


    def forward(self, input_tensor, motion_maps, ego_motions, rgb_cam, seg_cam, dep_cam, direction):

        reuse = False
        pred = []
        trans_pred = []
        pre_trans_pred = []
        rgb_pred = []
        seg_pred = []
        dep_pred = []
        trans_pred = []
        dir_pred = []
        speedyaw_pred = []
        for iter_index in range(self.iterations):
            print("Iteration " + str(iter_index))
            # Ground Truth as Input
            for t in range(self.K):
                timestep = iter_index * (self.K + self.T) + t
                #gridmap part
                motion_enc_input = tf.concat([input_tensor[:, :, :, timestep, :]], axis=3)

                # extract transformation matrix
                transform_matrix = ego_motions[:,timestep]
                h_motion, res_m = self.motion_enc(motion_enc_input, reuse=reuse)

                # do not need posterior for evaluation
                h_motion, _ = self.dense_block(h_motion, reuse=reuse)
                decoded_output = self.dec_cnn(h_motion, res_m, reuse=reuse)
                pre_trans_pred.append(tf.identity(decoded_output))
                prediction_output = decoded_output
                prediction_output = spatial_transformer_network((prediction_output + 1) / 2, transform_matrix[:,0,:6]) * 2 - 1
                trans_pred.append(tf.identity(prediction_output))
                self.transform_hidden_states(transform_matrix)
                pred.append(prediction_output)

                # image channels
                with tf.variable_scope("CAM", reuse=reuse):
                    cell, res = self.conv_data(rgb_cam[:,timestep], seg_cam[:,timestep], dep_cam[:,timestep], direction[:, timestep], motion_enc_input, transform_matrix, reuse)
                    p_rgb, p_seg, p_dep, _, p_dir, p_sy = self.deconv_data(cell, res, reuse)
                    rgb_pred.append(p_rgb)
                    seg_pred.append(p_seg)
                    dep_pred.append(p_dep)
                    trans_pred.append(tf.stack([self.transMat(p_sy[i]) for i in range(self.batch_size)]))
                    dir_pred.append(p_dir)
                    speedyaw_pred.append(p_sy)
                reuse = True

            # Prediction sequence
            for t in range(self.T):
                timestep = iter_index * (self.K + self.T) + self.K + t

                #gridmap part, use previous prediction for next prediction
                motion_enc_input = tf.concat([self.keep_alive(pred[-1]), self.pred_occlusion_map], axis=3)
                # comment this line to use last gt transformation matrix
                transform_matrix = trans_pred[-1]
                h_motion, res_m = self.motion_enc(motion_enc_input, reuse=reuse)
                h_motion, posterior = self.dense_block(h_motion, reuse=reuse)
                decoded_output = self.dec_cnn(h_motion, res_m, reuse=reuse)
                pre_trans_pred.append(tf.identity(decoded_output))
                prediction_output = decoded_output
                prediction_output = spatial_transformer_network((prediction_output + 1) / 2, transform_matrix[:,0,:6]) * 2 - 1
                trans_pred.append(tf.identity(prediction_output))
                self.transform_hidden_states(transform_matrix)
                pred.append(prediction_output)

                # image channels, reuse previous prediction
                with tf.variable_scope("CAM", reuse=reuse):
                    cell, res = self.conv_data(rgb_pred[-1], seg_pred[-1], dep_pred[-1], dir_pred[-1], motion_enc_input, transform_matrix, reuse)
                    p_rgb, p_seg, p_dep, _, p_dir, p_sy = self.deconv_data(cell, res, reuse)
                    rgb_pred.append(p_rgb)
                    seg_pred.append(p_seg)
                    dep_pred.append(p_dep)
                    trans_pred.append(tf.stack([self.transMat(p_sy[i]) for i in range(self.batch_size)]))
                    dir_pred.append(p_dir)
                    speedyaw_pred.append(p_sy)

        return pred, trans_pred, pre_trans_pred, rgb_pred, seg_pred, dep_pred, trans_pred, dir_pred, speedyaw_pred

    def conv_data(self, rgb, seg, dep, direction, gridmap, transform_matrix, reuse):
        #insipred by SNA model from https://arxiv.org/pdf/1710.05268.pdf
        res_in = []

        # first reduction
        rgb_conv1 = relu(conv2d(rgb, output_dim=self.df_dim, k_h=8, k_w=8, d_h=2, d_w=2, name='rgb_conv1', reuse=reuse), useSELU=self.useSELU)
        seg_conv1 = relu(conv2d(seg, output_dim=self.gf_dim, k_h=8, k_w=8, d_h=2, d_w=2, name='seg_conv1', reuse=reuse), useSELU=self.useSELU)
        dep_conv1 = relu(conv2d(dep, output_dim=self.gf_dim, k_h=8, k_w=8, d_h=2, d_w=2, name='dep_conv1', reuse=reuse), useSELU=self.useSELU)
        res_in.append([rgb_conv1, seg_conv1, dep_conv1])

        rgb_conv2 = relu(conv2d(rgb_conv1, output_dim=self.df_dim, k_h=4, k_w=4, d_h=1, d_w=1, name='rgb_conv2', reuse=reuse), useSELU=self.useSELU)
        seg_conv2 = relu(conv2d(seg_conv1, output_dim=self.gf_dim, k_h=4, k_w=4, d_h=1, d_w=1, name='seg_conv2', reuse=reuse), useSELU=self.useSELU)
        dep_conv2 = relu(conv2d(dep_conv1, output_dim=self.gf_dim, k_h=4, k_w=4, d_h=1, d_w=1, name='dep_conv2', reuse=reuse), useSELU=self.useSELU)

        # first LSTM
        motion_rgb_1 = BasicConvLSTMCell([self.data_size[0] // 2, self.data_size[1] // 2], [3, 3], self.gf_dim)
        if not reuse:
            self.motion_rgb_state_1 = tf.zeros([self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim*2])
        h_rgb_1, self.motion_rgb_state_1 = motion_rgb_1(rgb_conv2, self.motion_rgb_state_1, scope="Motion_LSTM_RGB_1", reuse=reuse)

        motion_seg_1 = BasicConvLSTMCell([self.data_size[0] // 2, self.data_size[1] // 2], [3, 3], self.gf_dim//2)
        if not reuse:
            self.motion_seg_state_1 = tf.zeros([self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim])
        h_seg_1, self.motion_seg_state_1 = motion_seg_1(seg_conv2, self.motion_seg_state_1, scope="Motion_LSTM_SEG_1", reuse=reuse)

        motion_dep_1 = BasicConvLSTMCell([self.data_size[0] // 2, self.data_size[1] // 2], [3, 3], self.gf_dim//2)
        if not reuse:
            self.motion_dep_state_1 = tf.zeros([self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim])
        h_dep_1, self.motion_dep_state_1 = motion_dep_1(dep_conv2, self.motion_dep_state_1, scope="Motion_LSTM_DEP_1", reuse=reuse)

        # second reduction
        rgb_conv3 = relu(conv2d(h_rgb_1, output_dim=self.gf_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='rgb_conv3', reuse=reuse), useSELU=self.useSELU)
        seg_conv3 = relu(conv2d(h_seg_1, output_dim=self.gf_dim//2, k_h=3, k_w=3, d_h=2, d_w=2, name='seg_conv3', reuse=reuse), useSELU=self.useSELU)
        dep_conv3 = relu(conv2d(h_dep_1, output_dim=self.gf_dim//2, k_h=3, k_w=3, d_h=2, d_w=2, name='dep_conv3', reuse=reuse), useSELU=self.useSELU)
        res_in.append([rgb_conv3, seg_conv3, dep_conv3])

        rgb_conv4 = relu(conv2d(rgb_conv3, output_dim=self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='rgb_conv4', reuse=reuse), useSELU=self.useSELU)
        seg_conv4 = relu(conv2d(seg_conv3, output_dim=self.gf_dim//2, k_h=3, k_w=3, d_h=1, d_w=1, name='seg_conv4', reuse=reuse), useSELU=self.useSELU)
        dep_conv4 = relu(conv2d(dep_conv3, output_dim=self.gf_dim//2, k_h=3, k_w=3, d_h=1, d_w=1, name='dep_conv4', reuse=reuse), useSELU=self.useSELU)

        # second LSTM, current shape rgb [batch_size, 20, 60, 32], seg/dep [batch_size, 20, 60, 16]
        motion_rgb_2 = BasicConvLSTMCell([self.data_size[0] // 8, self.data_size[1] // 8], [3, 3], self.gf_dim)
        if not reuse:
            self.motion_rgb_state_2 = tf.zeros([self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, self.gf_dim*2])
        h_rgb_2, self.motion_rgb_state_2 = motion_rgb_2(rgb_conv4, self.motion_rgb_state_2, scope="Motion_LSTM_RGB_2", reuse=reuse)

        motion_seg_2 = BasicConvLSTMCell([self.data_size[0] // 8, self.data_size[1] // 8], [3, 3], self.gf_dim//2)
        if not reuse:
            self.motion_seg_state_2 = tf.zeros([self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, self.gf_dim])
        h_seg_2, self.motion_seg_state_2 = motion_seg_2(seg_conv4, self.motion_seg_state_2, scope="Motion_LSTM_SEG_2", reuse=reuse)

        motion_dep_2 = BasicConvLSTMCell([self.data_size[0] // 8, self.data_size[1] // 8], [3, 3], self.gf_dim//2)
        if not reuse:
            self.motion_dep_state_2 = tf.zeros([self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, self.gf_dim])
        h_dep_2, self.motion_dep_state_2 = motion_dep_2(dep_conv4, self.motion_dep_state_2, scope="Motion_LSTM_DEP_2", reuse=reuse)

        # last reduction, concatenate with auxiliary info (grid map, transformation matrix, direction)
        rgb_conv5 = relu(conv2d(h_rgb_2, output_dim=self.df_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='rgb_conv5', reuse=reuse), useSELU=self.useSELU)
        seg_conv5 = relu(conv2d(h_seg_2, output_dim=self.gf_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='seg_conv5', reuse=reuse), useSELU=self.useSELU)
        dep_conv5 = relu(conv2d(h_dep_2, output_dim=self.gf_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='dep_conv5', reuse=reuse), useSELU=self.useSELU)

        imgs_concat = tf.concat([rgb_conv5, seg_conv5, dep_conv5],3)
        #add additional info
        transform_layer1 = tf.layers.dense(transform_matrix, self.df_dim, tf.nn.selu)
        transform_layer2 = tf.layers.dense(transform_layer1, (self.data_size[0] // 4) * (self.data_size[1] // 4) // 3, tf.nn.selu)
        transform_reshape = tf.reshape(transform_layer2,[self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, 1])
        motion_transform = BasicConvLSTMCell([self.data_size[0] // 8, self.data_size[1] // 8], [3, 3], 1)
        if not reuse:
            self.motion_transform_state = tf.zeros([self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, 2])
        motion_transform_cell, self.motion_transform_state = motion_transform(transform_reshape, self.motion_transform_state, scope="Motion_transform", reuse=reuse)
        transform_conv = relu(conv2d(motion_transform_cell, output_dim=1, k_h=3, k_w=3, d_h=2, d_w=2, name='transform_conv', reuse=reuse), useSELU=self.useSELU)

        direction_layer1 = tf.layers.dense(tf.cast(direction, tf.float32), self.df_dim, tf.nn.selu)
        direction_dropout1 = tf.layers.dropout(direction_layer1,0.75)
        direction_layer2 = tf.layers.dense(direction_dropout1, (self.data_size[0] // 4) * (self.data_size[1] // 4), tf.nn.selu)
        direction_dropout2 = tf.layers.dropout(direction_layer2,0.75)
        direction_reshape = tf.reshape(direction_dropout2,[self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, 1])
        direction_transform = BasicConvLSTMCell([self.data_size[0] // 8, self.data_size[1] // 8], [3, 3], 1)
        if not reuse:
            self.direction_transform_state = tf.zeros([self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, 2])
        direction_transform_cell, self.direction_transform_state = direction_transform(direction_reshape, self.direction_transform_state, scope="Direction_transform", reuse=reuse)
        direction_conv = relu(conv2d(direction_transform_cell, output_dim=1, k_h=3, k_w=3, d_h=2, d_w=2, name='direction_conv', reuse=reuse), useSELU=self.useSELU)

        gridmap_conv = relu(conv2d(gridmap, output_dim=self.gf_dim, k_h=3, k_w=3, d_h=2, d_w=2, name='gridmap_conv', reuse=reuse), useSELU=self.useSELU)
        gridmap_transform = BasicConvLSTMCell([self.image_size[0]//2, self.image_size[1]//2], [3, 3], self.gf_dim//2)
        if not reuse:
            self.gridmap_transform_state = tf.zeros([self.batch_size, self.image_size[0]//2, self.image_size[1]//2, self.gf_dim])
        gridmap_transform_cell, self.gridmap_transform_state = gridmap_transform(gridmap_conv, self.gridmap_transform_state, scope="Gridmap_transform", reuse=reuse)
        gridmap_conv2 = relu(conv2d(gridmap_transform_cell, output_dim=self.gf_dim//2, k_h=3, k_w=3, d_h=2, d_w=2, name='gridmap_conv2', reuse=reuse), useSELU=self.useSELU)
        gridmap_reshape1 = tf.reshape(gridmap_conv2, [self.batch_size, self.image_size[0]//4 * self.image_size[1]//4 * self.gf_dim//2])
        gridmap_dense = tf.layers.dense(gridmap_reshape1, (self.data_size[0] // 8) * (self.data_size[1] // 8), tf.nn.selu)
        gridmap_reshape2 = tf.reshape(gridmap_dense, [self.batch_size, self.data_size[0] // 8, self.data_size[1] // 8, 1])

        imgs_concat = tf.concat([imgs_concat, gridmap_reshape2, transform_conv, direction_conv],3)

        #reduce number of filters from 128+3 to 32
        conv6 = relu(conv2d(imgs_concat, output_dim=self.gf_dim, k_h=1, k_w=1, d_h=1, d_w=1, name='conv6', reuse=reuse), useSELU=self.useSELU)
        conv7 = relu(conv2d(conv6, output_dim=self.gf_dim, k_h=3, k_w=3, d_h=1, d_w=1, name='conv8', reuse=reuse), useSELU=self.useSELU)

        # LSTM before VAE/dense block
        motion = BasicConvLSTMCell([self.data_size[0] // 8, self.data_size[1] // 8], [3, 3], self.df_dim)
        if not reuse:
            self.motion_state = tf.zeros([self.batch_size, self.data_size[0] // 8, self.data_size[1] // 8, self.df_dim])
        ret_cell, self.motion_state = motion_rgb_1(conv7, self.motion_state, scope="Motion", reuse=reuse)

        return ret_cell, res_in

    def deconv_data(self, cell, res, reuse):
        if not self.useDenseBlock:
            # VAE stage
            with tf.variable_scope("scale_deconv", reuse=reuse):
                scale_deconv = tf.layers.dense(cell, self.gf_dim, tf.nn.softplus)
            latent = tf.contrib.distributions.MultivariateNormalDiag(cell, scale_deconv)
            logit = latent.sample(self.samples)
            logit = tf.reshape(logit, [-1] + cell.shape.as_list())
            cell = tf.reduce_mean(logit, axis=0)
        else:
            latent = 0

        # first deconv stage
        deconv_latent = BasicConvLSTMCell([self.data_size[0] // 4, self.data_size[1] // 4], [3, 3], self.gf_dim)
        if not reuse:
            self.deconv_latent_state = tf.zeros([self.batch_size, self.data_size[0] // 8, self.data_size[1] // 8, self.gf_dim*2])
        de_mot_latent, self.deconv_latent_state = deconv_latent(cell, self.deconv_latent_state, scope="deconv_latent", reuse=reuse)

        out_shape1 = [self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, self.df_dim * 2]
        deconv1 = relu(deconv2d(de_mot_latent, output_shape=out_shape1, k_h=3, k_w=3, d_h=2, d_w=2, name='dec_deconv1', reuse=reuse), useSELU=self.useSELU)

        # splitting into separate channels
        rgb_deconv1 = deconv1[:,:,:,:self.df_dim]
        seg_deconv1 = deconv1[:,:,:,self.df_dim:self.df_dim+self.gf_dim]
        dep_deconv1 = deconv1[:,:,:,self.df_dim+self.gf_dim:-3]

        # new version training transformation and direction
        transformation_deconv1 = tf.layers.flatten(deconv1[:,:,:,-2])
        direction_deconv1 = tf.layers.flatten(deconv1[:,:,:,-1])
        tl1 = tf.layers.dense(transformation_deconv1, self.gf_dim//2, tf.nn.selu)
        tl1_dropout = tf.layers.dropout(tl1,0.5)
        tl2 = tf.layers.dense(tl1_dropout, self.gf_dim//4, tf.nn.selu)
        tl3 = tf.layers.dense(tl2, self.gf_dim//8, tf.nn.selu)

        dir1 = tf.layers.dense(direction_deconv1, self.gf_dim//2, tf.nn.selu)
        dir2 = tf.layers.dense(dir1, 1*2, tf.nn.sigmoid)
        dir_out = tf.reshape(dir2, [self.batch_size,2])

        #first value contains vel, second yaw_rate, multiply by 100 & 20 for tanh -1,1 range extension
        sy_out = tf.concat([tf.layers.dense(tl3, 1, tf.nn.tanh)*100, tf.layers.dense(tl3, 1, tf.nn.tanh)*20], axis=1)

        # second deconv stage
        out_shape2 = [self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, self.gf_dim]
        deconv_rgb_1 = relu(deconv2d(rgb_deconv1, output_shape=out_shape2, k_h=3, k_w=3, d_h=1, d_w=1, name='deconv_rgb_1', reuse=reuse), useSELU=self.useSELU)
        deconv_seg_1 = relu(deconv2d(seg_deconv1, output_shape=out_shape2, k_h=3, k_w=3, d_h=1, d_w=1, name='deconv_seg_1', reuse=reuse), useSELU=self.useSELU)
        deconv_dep_1 = relu(deconv2d(dep_deconv1, output_shape=out_shape2, k_h=3, k_w=3, d_h=1, d_w=1, name='deconv_dep_1', reuse=reuse), useSELU=self.useSELU)

        deconv_motion_rgb = BasicConvLSTMCell([self.data_size[0] // 4, self.data_size[1] // 4], [3, 3], self.gf_dim)
        if not reuse:
            self.deconv_motion_rgb_state = tf.zeros([self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, self.gf_dim*2])
        de_mot_rgb, self.deconv_motion_rgb_state = deconv_motion_rgb(rgb_deconv1, self.deconv_motion_rgb_state, scope="deconv_motion_rgb", reuse=reuse)

        deconv_motion_seg = BasicConvLSTMCell([self.data_size[0] // 4, self.data_size[1] // 4], [3, 3], self.gf_dim)
        if not reuse:
            self.deconv_motion_seg_state = tf.zeros([self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, self.gf_dim*2])
        de_mot_seg, self.deconv_motion_seg_state = deconv_motion_seg(seg_deconv1, self.deconv_motion_seg_state, scope="deconv_motion_seg", reuse=reuse)

        deconv_motion_dep = BasicConvLSTMCell([self.data_size[0] // 4, self.data_size[1] // 4], [3, 3], self.gf_dim)
        if not reuse:
            self.deconv_motion_dep_state = tf.zeros([self.batch_size, self.data_size[0] // 4, self.data_size[1] // 4, self.gf_dim*2])
        de_mot_dep, self.deconv_motion_dep_state = deconv_motion_dep(dep_deconv1, self.deconv_motion_dep_state, scope="deconv_motion_dep", reuse=reuse)

        # concatenate with residual connections
        de_mot_rgb = tf.concat([de_mot_rgb, res[1][0]], 3)
        de_mot_seg = tf.concat([de_mot_seg, res[1][1]], 3)
        de_mot_dep = tf.concat([de_mot_dep, res[1][2]], 3)

        # second deconv stage
        out_shape_rgb_1 = [self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim]
        out_shape_seg_1 = [self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim//2]
        out_shape_dep_1 = [self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim//2]
        deconv_rgb_2 = relu(deconv2d(de_mot_rgb, output_shape=out_shape_rgb_1, k_h=3, k_w=3, d_h=2, d_w=2, name='deconv_rgb_2', reuse=reuse), useSELU=self.useSELU)
        deconv_seg_2 = relu(deconv2d(de_mot_seg, output_shape=out_shape_seg_1, k_h=3, k_w=3, d_h=2, d_w=2, name='deconv_seg_2', reuse=reuse), useSELU=self.useSELU)
        deconv_dep_2 = relu(deconv2d(de_mot_dep, output_shape=out_shape_dep_1, k_h=3, k_w=3, d_h=2, d_w=2, name='deconv_dep_2', reuse=reuse), useSELU=self.useSELU)

        out_shape_rgb_2 = [self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim//2]
        out_shape_seg_2 = [self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim//4]
        out_shape_dep_2 = [self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim//4]
        deconv_rgb_3 = relu(deconv2d(deconv_rgb_2, output_shape=out_shape_rgb_2, k_h=3, k_w=3, d_h=1, d_w=1, name='deconv_rgb_3', reuse=reuse), useSELU=self.useSELU)
        deconv_seg_3 = relu(deconv2d(deconv_seg_2, output_shape=out_shape_seg_2, k_h=3, k_w=3, d_h=1, d_w=1, name='deconv_seg_3', reuse=reuse), useSELU=self.useSELU)
        deconv_dep_3 = relu(deconv2d(deconv_dep_2, output_shape=out_shape_dep_2, k_h=3, k_w=3, d_h=1, d_w=1, name='deconv_dep_3', reuse=reuse), useSELU=self.useSELU)

        deconv_motion_rgb_2 = BasicConvLSTMCell([self.data_size[0] // 2, self.data_size[1] // 2], [3, 3], self.gf_dim//2)
        if not reuse:
            self.deconv_motion_rgb_2_state = tf.zeros([self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim])
        de_mot_rgb_2, self.deconv_motion_rgb_2_state = deconv_motion_rgb_2(deconv_rgb_3, self.deconv_motion_rgb_2_state, scope="deconv_motion_rgb_2", reuse=reuse)

        deconv_motion_seg_2 = BasicConvLSTMCell([self.data_size[0] // 2, self.data_size[1] // 2], [3, 3], self.gf_dim//4)
        if not reuse:
            self.deconv_motion_seg_2_state = tf.zeros([self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim//2])
        de_mot_seg_2, self.deconv_motion_seg_2_state = deconv_motion_seg_2(deconv_seg_3, self.deconv_motion_seg_2_state, scope="deconv_motion_seg_2", reuse=reuse)

        deconv_motion_dep_2 = BasicConvLSTMCell([self.data_size[0] // 2, self.data_size[1] // 2], [3, 3], self.gf_dim//4)
        if not reuse:
            self.deconv_motion_dep_2_state = tf.zeros([self.batch_size, self.data_size[0] // 2, self.data_size[1] // 2, self.gf_dim//2])
        de_mot_dep_2, self.deconv_motion_dep_2_state = deconv_motion_dep_2(deconv_dep_3, self.deconv_motion_dep_2_state, scope="deconv_motion_dep_2", reuse=reuse)

        # concatenate with residual connections
        de_mot_rgb_2 = tf.concat([de_mot_rgb_2, res[0][0]], 3)
        de_mot_seg_2 = tf.concat([de_mot_seg_2, res[0][1]], 3)
        de_mot_dep_2 = tf.concat([de_mot_dep_2, res[0][2]], 3)

        # third deconv stage
        out_shape_rgb_3 = [self.batch_size, self.data_size[0], self.data_size[1], self.gf_dim//2]
        out_shape_seg_3 = [self.batch_size, self.data_size[0], self.data_size[1], self.gf_dim//4]
        out_shape_dep_3 = [self.batch_size, self.data_size[0], self.data_size[1], self.gf_dim//4]
        deconv_rgb_4 = relu(deconv2d(de_mot_rgb_2, output_shape=out_shape_rgb_3, k_h=3, k_w=3, d_h=2, d_w=2, name='deconv_rgb_4', reuse=reuse), useSELU=self.useSELU)
        deconv_seg_4 = relu(deconv2d(de_mot_rgb_2, output_shape=out_shape_seg_3, k_h=3, k_w=3, d_h=2, d_w=2, name='deconv_seg_4', reuse=reuse), useSELU=self.useSELU)
        deconv_dep_4 = relu(deconv2d(de_mot_rgb_2, output_shape=out_shape_dep_3, k_h=3, k_w=3, d_h=2, d_w=2, name='deconv_dep_4', reuse=reuse), useSELU=self.useSELU)

        out_shape_rgb_4 = [self.batch_size, self.data_size[0], self.data_size[1], 3]
        out_shape_seg_4 = [self.batch_size, self.data_size[0], self.data_size[1], 3]
        out_shape_dep_4 = [self.batch_size, self.data_size[0], self.data_size[1], 1]
        deconv_rgb_5 = tanh(deconv2d(deconv_rgb_4, output_shape=out_shape_rgb_4, k_h=3, k_w=3, d_h=1, d_w=1, name='deconv_rgb_5', reuse=reuse))
        deconv_seg_5 = tanh(deconv2d(deconv_seg_4, output_shape=out_shape_seg_4, k_h=3, k_w=3, d_h=1, d_w=1, name='deconv_seg_5', reuse=reuse))
        deconv_dep_5 = tanh(deconv2d(deconv_dep_4, output_shape=out_shape_dep_4, k_h=3, k_w=3, d_h=1, d_w=1, name='deconv_dep_5', reuse=reuse))

        return deconv_rgb_5, deconv_seg_5, deconv_dep_5, latent, dir_out, sy_out

    # encoder stage of grid map model
    def motion_enc(self, motion_in, reuse):
        res_in = []
        conv1_1 = relu(conv2d(motion_in, output_dim=self.gf_dim, k_h=5, k_w=5, d_h=2, d_w=2, name='mot_conv1_1', reuse=reuse), useSELU=self.useSELU)
        res_in.append(conv1_1)

        motion_cell_1 = BasicConvLSTMCell([self.image_size[0] // 2, self.image_size[1] // 2], [3, 3], self.gf_dim)
        if not reuse:
            self.motion_cell_state_1 = tf.zeros([self.batch_size, self.image_size[0] // 2, self.image_size[1] // 2, self.gf_dim * 2])
        h_cell_1, self.motion_cell_state_1 = motion_cell_1(conv1_1, self.motion_cell_state_1, scope="Motion_LSTM_1", reuse=reuse)

        conv2_1 = relu(conv2d(h_cell_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,d_h=2, d_w=2, name='mot_conv2_1', reuse=reuse), useSELU=self.useSELU)

        res_in.append(conv2_1)

        motion_cell_2 = BasicConvLSTMCell([self.image_size[0] // 4, self.image_size[1] // 4],[3, 3], self.gf_dim * 2)
        if not reuse:
            self.motion_cell_state_2 = tf.zeros([self.batch_size, self.image_size[0] // 4,self.image_size[1] // 4, self.gf_dim * 4])
        h_cell_2, self.motion_cell_state_2 = motion_cell_2(conv2_1, self.motion_cell_state_2, scope="Motion_LSTM_2", reuse=reuse)

        return h_cell_2, res_in

    # dense block or VAE for grid map generation
    def dense_block(self, h_comb, reuse=False):
        if self.useDenseBlock:
            conv3_1 = relu(dilated_conv2d(h_comb, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                                                     dilation_rate=1, name='mot_dil_conv3_1', reuse=reuse), useSELU=self.useSELU)
            conv3_2 = relu(dilated_conv2d(conv3_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                                                     dilation_rate=2, name='mot_dil_conv3_2', reuse=reuse), useSELU=self.useSELU)
            return conv3_2, 0
        else:
            second_layer = relu(dilated_conv2d(h_comb, output_dim=self.gf_dim, k_h=3, k_w=3,dilation_rate=1, name='mot_dil_conv3_2', reuse=reuse), useSELU=self.useSELU)

            with tf.variable_scope("scale", reuse=reuse):
                scale = tf.layers.dense(second_layer, self.gf_dim, tf.nn.softplus)
            latent = tf.contrib.distributions.MultivariateNormalDiag(second_layer, scale)

            logit = latent.sample(self.samples)
            logit = tf.reshape(logit, [-1] + second_layer.shape.as_list())
            logit = tf.reduce_mean(logit, axis=0)

            return logit, latent

    # decoder stage of grid map model
    def dec_cnn(self, h_comb, res_connect, reuse=False):
        #new
        motion_cell_3 = BasicConvLSTMCell([self.image_size[0] // 4, self.image_size[1] // 4],
                                          [3, 3], self.gf_dim * 2)
        if not reuse:
            self.motion_cell_state_3 = tf.zeros([self.batch_size, self.image_size[0] // 4,
                                                 self.image_size[1] // 4, self.gf_dim * 4])
        h_comb, self.motion_cell_state_3 = motion_cell_3(
            h_comb, self.motion_cell_state_3, scope="Motion_LSTM_3", reuse=reuse)
        #---
        shapel2 = [self.batch_size, self.image_size[0] // 2,
                   self.image_size[1] // 2, self.gf_dim * 2]
        shapeout3 = [self.batch_size, self.image_size[0] // 2,
                     self.image_size[1] // 2, self.gf_dim]

        deconv2_concat = tf.concat(axis=3, values=[h_comb, res_connect[1]])
        deconv2_2 = relu(deconv2d(deconv2_concat,
                                             output_shape=shapel2, k_h=3, k_w=3,
                                             d_h=2, d_w=2, name='dec_deconv2_2', reuse=reuse), useSELU=self.useSELU)

        decoder_cell_1 = BasicConvLSTMCell([self.image_size[0] // 2, self.image_size[1] // 2],
                                          [3, 3], self.gf_dim * 2)
        if not reuse:
            self.decoder_cell_state_1 = tf.zeros([self.batch_size, self.image_size[0] // 2,
                                                 self.image_size[1] // 2, self.gf_dim * 4])
        h_cell_1, self.decoder_cell_state_1 = decoder_cell_1(
            deconv2_2, self.decoder_cell_state_1, scope="Decoder_LSTM_1", reuse=reuse)

        shapel1 = [self.batch_size, self.image_size[0],
                   self.image_size[1], self.gf_dim]
        shapeout1 = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim]

        deconv1_concat = tf.concat(axis=3, values=[h_cell_1, res_connect[0]])
        deconv1_2 = relu(deconv2d(deconv1_concat,
                                             output_shape=shapel1, k_h=3, k_w=3, d_h=2, d_w=2,
                                             name='dec_deconv1_2', reuse=reuse), useSELU=self.useSELU)
        xtp1 = tanh(deconv2d(deconv1_2, output_shape=shapeout1, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='dec_deconv1_1', reuse=reuse))
        return xtp1

    def motion_maps_combined(self, motion_maps, decoded_output, reuse=False):
        combined_input = tf.concat([motion_maps, decoded_output], axis=3)
        conv1 = relu(conv2d(combined_input, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='comb_conv1', reuse=reuse), useSELU=self.useSELU)
        conv2 = tanh(conv2d(conv1, output_dim=self.c_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='comb_conv2', reuse=reuse), useSELU=self.useSELU)
        return conv2

    def sharpen_image(self, decoded_output, future_maps, orig_input, reuse=False):
        combined_input = tf.concat([decoded_output, future_maps, orig_input], axis=3)
        if self.small_sharpener:
            conv2 = tanh(conv2d(combined_input, output_dim=self.c_dim, k_h=5, k_w=5,
                              d_h=1, d_w=1, name='sharpen_conv2', reuse=reuse))
        else:
            conv1 = relu(conv2d(combined_input, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='sharpen_conv1', reuse=reuse), useSELU=self.useSELU)
            conv2 = tanh(conv2d(conv1, output_dim=self.c_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='sharpen_conv2', reuse=reuse))
        return conv2

    def keep_alive(self, prediction, factor=2.5):
        return tanh(factor * prediction)

    def transform_hidden_states(self, transform_vector):
        self.motion_cell_state_1 = spatial_transformer_network(self.motion_cell_state_1, transform_vector[:,1,:6])
        self.motion_cell_state_2 = spatial_transformer_network(self.motion_cell_state_2, transform_vector[:,2,:6])
        self.motion_cell_state_3 = spatial_transformer_network(self.motion_cell_state_3, transform_vector[:,2,:6])
        self.decoder_cell_state_1 = spatial_transformer_network(self.decoder_cell_state_1, transform_vector[:,1,:6])

    def discriminator(self, image):
        h0 = relu(conv2d(image, output_dim=self.df_dim, k_h=3, k_w=3,
                                d_h=2, d_w=2, name='dis_h0_conv'), useSELU=self.useSELU)
        h1 = relu(dilated_conv2d(h0, output_dim=self.df_dim * 2, k_h=3, k_w=3,
                                        dilation_rate=1, name='dis_h1_conv'), useSELU=self.useSELU)
        h2 = relu(conv2d(h1, output_dim=self.df_dim * 2, k_h=3, k_w=3,
                                d_h=2, d_w=2, name='dis_h2_conv'), useSELU=self.useSELU)
        h3 = relu(dilated_conv2d(h2, output_dim=self.df_dim, k_h=3, k_w=3,
                                        dilation_rate=1, name='dis_h3_conv'), useSELU=self.useSELU)
        h = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'dis_h3_lin')

        return tf.nn.sigmoid(h), h

    def save(self, sess, checkpoint_dir, step):
        model_name = "MCNET.model"

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, sess, checkpoint_dir, model_name=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None:
                model_name = ckpt_name
            try:
                self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            except:
                try:
                    path = os.path.join(checkpoint_dir, model_name)+'/MCNET.model-80002'
                    print(path)
                    saver = tf.train.import_meta_graph(path+'.meta')
                    saver.restore(sess, path)
                except:
                    print("FAILED: could not load any model checkpoint or model files directly.")
                    return False, ckpt
            return True, ckpt
        else:
            return False, ckpt

    def add_input_to_generated_data(self, generated_data, input_data):
        combined_data = []
        for iter_index in range(self.iterations):
            start_frame_input = iter_index * (self.K + self.T)
            end_frame_input = iter_index * (self.K + self.T) + self.K
            combined_data.append(
                input_data[:, :, :, start_frame_input:end_frame_input, :self.c_dim])
            start_frame_generated = iter_index * self.T
            end_frame_generated = (iter_index + 1) * self.T
            combined_data.append(
                generated_data[:, :, :, start_frame_generated:end_frame_generated, :])
        return tf.concat(combined_data, axis=3)

    def mask_black(self, data_to_mask, occlusion_mask):
        data_to_mask = (data_to_mask + 1) / 2.0
        data_to_mask = tf.multiply(data_to_mask, occlusion_mask)
        data_to_mask = (data_to_mask * 2.0) - 1
        return data_to_mask

    def weighted_BCE_loss(self, predictions, labels, weight0=1, weight1=1):
        with tf.variable_scope("WCE_Loss"):
            predictions = (predictions + 1) / 2.0
            labels = (labels + 1) / 2.0
            labels = tf.clip_by_value(labels, 0, 1)
            epsilon = tf.constant(1e-10, dtype=tf.float32, name="epsilon")
            predictions = tf.clip_by_value(predictions, 0, 1-epsilon)
            coef0 = labels * weight0
            coef1 = (1 - labels) * weight1
            coefficient = coef0 + coef1
            label_shape = labels.get_shape().as_list()
            coefficient = tf.ones(label_shape)
            self.max_pred = tf.reduce_max(predictions)
            self.min_pred = tf.reduce_min(predictions)
            self.max_labels = tf.reduce_max(labels)
            self.min_labels = tf.reduce_min(labels)
            print(label_shape)
            cross_entropy = - 1.0 / label_shape[0] / label_shape[3] * \
                tf.reduce_sum(tf.multiply(
                    labels * tf.log(predictions + epsilon) +
                    (1 - labels) * tf.log(1 - predictions + epsilon), coefficient))
            cross_entropy_mean = tf.reduce_mean(
                cross_entropy, name="cross_entropy")
            return cross_entropy_mean

    # from https://stackoverflow.com/questions/48707974/deg2rad-conversion-in-tensorflow
    def tf_deg2rad(self, deg):
        pi_on_180 = 0.017453292519943295
        return deg * pi_on_180

    # from preprocessing script
    def transMat(self, speedyawrate):
        period_duration = 1.0 / self.frame_rate
        yaw_diff = self.tf_deg2rad(speedyawrate[1] * period_duration)
        tMat = []
        for i in range(3): # shape has to be [3,8]
            imgsize = self.image_size[0] // (2 ** i)
            pixel_size = self.gridmap_size * 1.0 / imgsize # [m]
            pixel_diff = speedyawrate[0] * period_duration * 1.0 / pixel_size
            dx = tf.cos(yaw_diff) * pixel_diff
            dy = tf.sin(yaw_diff) * pixel_diff
            theta = -yaw_diff
            a11 = tf.cos(theta)
            a12 = -tf.sin(theta)
            a13 = dx / ((imgsize - 1) / 2.0)
            a21 = tf.sin(theta)
            a22 = tf.cos(theta)
            a23 = dy / ((imgsize - 1) / 2.0)
            tMat.append([a11, a12, a13, a21, a22, a23, 0, 0])
        return tf.stack(tMat)
