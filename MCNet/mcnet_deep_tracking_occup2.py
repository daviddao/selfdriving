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


class MCNET(object):

    def __init__(self, image_size, batch_size=32, c_dim=1,  # Input has to be 2 dimensional: First occupancy, last occlusion map
                 K=10, T=10, checkpoint_dir=None, is_train=True,
                 iterations=1, d_input_frames=20):

        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train
        self.d_input_frames = d_input_frames

        self.gf_dim = 64
        self.df_dim = 64

        self.c_dim = c_dim
        self.K = K
        self.T = T
        self.iterations = iterations
        self.motion_shape = [batch_size, self.image_size[0],
                             self.image_size[1], iterations * (K + T), c_dim + 1]
        self.content_shape = [batch_size, self.image_size[0],
                              self.image_size[1], iterations * (K + T), c_dim + 1]
        self.target_shape = [batch_size, self.image_size[0],
                             self.image_size[1], iterations * (K + T), c_dim * 2 + 1]  # In here also the occlusion map for cutting out the gradients
        self.occlusion_shape = [batch_size, self.image_size[0],
                                self.image_size[1], 1]
        print "Target shape = " + str(self.target_shape)
        print "K = " + str(K) + ", T = " + str(T)
        print self.motion_shape
        self.build_model()

    def build_model(self):
        self.motion_in = tf.placeholder(
            tf.float32, self.motion_shape, name='motion_in')
        self.content_in = tf.placeholder(
            tf.float32, self.content_shape, name='content_in')
        self.target = tf.placeholder(
            tf.float32, self.target_shape, name='target')
        self.pred_occlusion_map = tf.ones(
            self.occlusion_shape, dtype=tf.float32, name='Pred_Occlusion_Map') * -1

        pred = self.forward(self.motion_in, self.content_in)
        self.G = tf.stack(axis=3, values=pred)

        if self.is_train:

            self.loss_occlusion_mask = (tf.tile(
                self.target[:, :, :, :, -1:], [1, 1, 1, 1, self.c_dim * 2]) + 1) / 2.0
            self.target_masked = self.mask_black(
                self.target[:, :, :, :, :self.c_dim * 2], self.loss_occlusion_mask)
            self.G_masked = self.mask_black(self.G, self.loss_occlusion_mask)

            """
            true_sim = inverse_transform(
                self.target_masked[:, :, :, :, :])  # was: [:,:,:,self.K:,:]
            if self.c_dim == 1:
                true_sim = tf.tile(true_sim, [1, 1, 1, 1, 3])
            true_sim = tf.reshape(tf.transpose(true_sim, [0, 3, 1, 2, 4]),
                                  [-1, self.image_size[0],
                                   self.image_size[1], 3])
            gen_sim = inverse_transform(self.G_masked)
            if self.c_dim == 1:
                gen_sim = tf.tile(gen_sim, [1, 1, 1, 1, 3])
            gen_sim = tf.reshape(tf.transpose(gen_sim, [0, 3, 1, 2, 4]),
                                 [-1, self.image_size[0],
                                  self.image_size[1], 3])
            """
            """
            binput = tf.reshape(self.target_masked[:, :, :, :, :],
                                [self.batch_size, self.image_size[0],
                                 self.image_size[1], -1])
            btarget = tf.reshape(self.target_masked[:, :, :, :, :],
                                 [self.batch_size, self.image_size[0],
                                  self.image_size[1], -1])
            bgen = tf.reshape(self.G_masked, [self.batch_size,
                                              self.image_size[0],
                                              self.image_size[1], -1])
            """
            """
            gen_seq = self.add_input_to_generated_data(self.G_masked, self.target_masked)
            self.gen_seq_masked = self.mask_black(gen_seq, input_occlusion_map)
            """
            """
            # TODO: try to get the random start point running
            d_iteration_index = tf.random_uniform([], minval=0, maxval=self.iterations - (
                self.d_input_frames / (self.K + self.T)), dtype=tf.int32)
            start_frame = d_iteration_index * (self.K + self.T)
            end_frame = start_frame + self.d_input_frames
            """

            start_frame = 0
            center_frame = self.d_input_frames / 2
            end_frame = self.d_input_frames
            gen_sequence = tf.concat(axis=3, values=[self.target_masked[
                                     :, :, :, start_frame:center_frame, :], self.G_masked[:, :, :, center_frame:end_frame, :]])
            gt_sequence = self.target_masked[:, :, :, start_frame:end_frame, :]
            good_data = tf.reshape(gt_sequence,
                                   [self.batch_size, self.image_size[0],
                                    self.image_size[1], -1])
            gen_data = tf.reshape(gen_sequence,
                                  [self.batch_size, self.image_size[0],
                                   self.image_size[1], -1])

            #good_data = tf.concat(axis=3,values=[binput,btarget])
            #gen_data  = tf.concat(axis=3,values=[binput,bgen])

            with tf.variable_scope("DIS", reuse=False):
                self.D, self.D_logits = self.discriminator(good_data)

            with tf.variable_scope("DIS", reuse=True):
                self.D_, self.D_logits_ = self.discriminator(gen_data)

            self.L_p = tf.reduce_mean(
                tf.square(self.G_masked - self.target_masked)
            )
            # self.L_gdl = gdl(gen_sim, true_sim, 1.)
            # self.L_img = self.L_p + self.L_gdl
            self.L_BCE = self.weighted_BCE_loss(
                self.G_masked, self.target_masked)
            self.L_img = self.L_BCE

            # Standard loss for real and fake (only for display and parameter
            # purpose, no loss trained on)
            self.d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits, labels=tf.ones_like(self.D)
                )
            )
            self.d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.zeros_like(self.D_)
                )
            )
            """
            self.d_loss = self.d_loss_real + self.d_loss_fake
            self.L_GAN = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.D_logits_, labels=tf.ones_like(self.D_)
                )
            )
            """
            #==================================================================
            # WGAN-GP
            LAMBDA = 10

            self.L_GAN = -tf.reduce_mean(self.D_)
            """
            self.d_loss = tf.reduce_mean(self.D_) - tf.reduce_mean(self.D)

            alpha = tf.random_uniform(
                shape=[self.batch_size, 1, 1, 1],
                minval=0.,
                maxval=1.
            )
            tile_shape = [1, self.image_size[0], self.image_size[
                1], self.d_input_frames * self.c_dim]
            alpha = tf.tile(alpha, tile_shape)

            differences = gen_data - good_data
            interpolates = good_data + (alpha * differences)
            interpolates = tf.reshape(interpolates, [self.batch_size, -1])
            discriminator_prediction_WGAN, _ = self.discriminator(tf.reshape(interpolates, [self.batch_size, self.image_size[
                                                                  0], self.image_size[1], self.d_input_frames * self.c_dim]))
            gradients = tf.gradients(tf.reshape(
                discriminator_prediction_WGAN, [-1]), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(
                tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.)**2)
            self.d_loss += LAMBDA * gradient_penalty
            #==================================================================
            """
            self.d_loss = self.d_loss_fake + self.d_loss_real

            self.loss_sum = tf.summary.scalar("L_img", self.L_img)
            self.L_p_sum = tf.summary.scalar("L_p", self.L_p)
            self.L_BCE_sum = tf.summary.scalar("L_BCE", self.L_BCE)
            self.L_GAN_sum = tf.summary.scalar("L_GAN", self.L_GAN)
            self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)
            self.d_loss_real_sum = tf.summary.scalar(
                "d_loss_real", self.d_loss_real)
            self.d_loss_fake_sum = tf.summary.scalar(
                "d_loss_fake", self.d_loss_fake)

            self.t_vars = tf.trainable_variables()
            self.g_vars = [var for var in self.t_vars if 'DIS' not in var.name]
            self.d_vars = [var for var in self.t_vars if 'DIS' in var.name]
            num_param = 0.0
            for var in self.g_vars:
                num_param += int(np.prod(var.get_shape()))
            print("Number of parameters: %d" % num_param)
        self.saver = tf.train.Saver(max_to_keep=10)

    def forward(self, motion_in, content_in):

        reuse = False
        pred = []
        for iter_index in range(self.iterations):
            print "Iteration " + str(iter_index)
            # Ground Truth as Input
            for t in range(self.K):
                timestep = iter_index * (self.K + self.T) + t
                content_enc_input = self.content_in[:, :, :, timestep, :]
                motion_enc_input = self.motion_in[:, :, :, timestep, :]
                h_content, res_c = self.content_enc(
                    content_enc_input, reuse=reuse)
                h_motion, res_m = self.motion_enc(
                    motion_enc_input, reuse=reuse)
                h_tp1 = self.comb_layers(h_motion, h_content, reuse=reuse)
                res_connect = self.residual(res_m, res_c, reuse=reuse)
                prediction_output = self.dec_cnn(
                    h_tp1, res_connect, reuse=reuse)
                pred.append(prediction_output)
                reuse = True

            # Prediction sequence
            for t in range(self.T):
                content_enc_input = tf.concat(
                    [pred[-1][:, :, :, 0:1], self.pred_occlusion_map], axis=3)
                motion_enc_input = tf.concat(
                    [pred[-1][:, :, :, 1:2], self.pred_occlusion_map], axis=3)
                h_content, res_c = self.content_enc(
                    content_enc_input, reuse=reuse)
                h_motion, res_m = self.motion_enc(
                    motion_enc_input, reuse=reuse)
                h_tp1 = self.comb_layers(h_motion, h_content, reuse=reuse)
                res_connect = self.residual(res_m, res_c, reuse=reuse)
                prediction_output = self.dec_cnn(
                    h_tp1, res_connect, reuse=reuse)
                pred.append(prediction_output)

        return pred

    def motion_enc(self, motion_in, reuse):
        res_in = []
        conv1_1 = relu(conv2d(motion_in, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='mot_conv1_1', reuse=reuse))
        conv1_2 = relu(batch_norm(conv2d(conv1_1, output_dim=self.gf_dim, k_h=3, k_w=3,
                                         d_h=1, d_w=1, name='mot_conv1_2', reuse=reuse), name='Motion_Conv1_2_Batchnorm', reuse=reuse))
        res_in.append(conv1_2)
        pool1 = MaxPooling(conv1_2, [2, 2])

        motion_cell_1 = BasicConvLSTMCell([self.image_size[0] / 2, self.image_size[1] / 2],
                                          [3, 3], self.gf_dim * 2)
        if not reuse:
            self.motion_cell_state_1 = tf.zeros([self.batch_size, self.image_size[0] / 2,
                                                 self.image_size[1] / 2, self.gf_dim * 4])
        h_cell_1, self.motion_cell_state_1 = motion_cell_1(
            pool1, self.motion_cell_state_1, scope="Motion_LSTM_1", reuse=reuse)

        conv2_1 = relu(batch_norm(dilated_conv2d(h_cell_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                                                 dilation_rate=1, name='mot_dil_conv2_1', reuse=reuse), name='Motion_Conv2_1_Batchnorm', reuse=reuse))
        conv2_2 = relu(batch_norm(dilated_conv2d(conv2_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                                                 dilation_rate=2, name='mot_dil_conv2_2', reuse=reuse), name='Motion_Conv2_2_Batchnorm', reuse=reuse))
        res_in.append(conv2_2)
        pool2 = MaxPooling(conv2_2, [2, 2])

        motion_cell_2 = BasicConvLSTMCell([self.image_size[0] / 4, self.image_size[1] / 4],
                                          [3, 3], self.gf_dim * 4)
        if not reuse:
            self.motion_cell_state_2 = tf.zeros([self.batch_size, self.image_size[0] / 4,
                                                 self.image_size[1] / 4, self.gf_dim * 8])
        h_cell_2, self.motion_cell_state_2 = motion_cell_2(
            pool2, self.motion_cell_state_2, scope="Motion_LSTM_2", reuse=reuse)

        conv3_1 = relu(batch_norm(dilated_conv2d(h_cell_2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                                                 dilation_rate=1, name='mot_dil_conv3_1', reuse=reuse), name='Motion_Conv3_1_Batchnorm', reuse=reuse))
        conv3_2 = relu(batch_norm(dilated_conv2d(conv3_1, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                                                 dilation_rate=2, name='mot_dil_conv3_2', reuse=reuse), name='Motion_Conv3_2_Batchnorm', reuse=reuse))
        conv3_3 = relu(batch_norm(dilated_conv2d(conv3_2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                                                 dilation_rate=4, name='mot_dil_conv3_3', reuse=reuse), name='Motion_Conv3_3_Batchnorm', reuse=reuse))
        res_in.append(conv3_3)
        pool3 = MaxPooling(conv3_3, [2, 2])

        motion_cell_3 = BasicConvLSTMCell([self.image_size[0] / 8, self.image_size[1] / 8],
                                          [3, 3], self.gf_dim * 4)
        if not reuse:
            self.motion_cell_state_3 = tf.zeros([self.batch_size, self.image_size[0] / 8,
                                                 self.image_size[1] / 8, self.gf_dim * 8])
        h_cell_3, self.motion_cell_state_3 = motion_cell_3(
            pool3, self.motion_cell_state_3, scope="Motion_LSTM_3", reuse=reuse)

        return h_cell_3, res_in

    def content_enc(self, xt, reuse):
        res_in = []
        conv1_1 = relu(conv2d(xt, output_dim=self.gf_dim, k_h=3, k_w=3,
                              d_h=1, d_w=1, name='cont_conv1_1', reuse=reuse))
        conv1_2 = relu(batch_norm(conv2d(conv1_1, output_dim=self.gf_dim, k_h=3, k_w=3,
                                         d_h=1, d_w=1, name='cont_conv1_2', reuse=reuse), name='Content_Conv1_2_Batchnorm', reuse=reuse))
        res_in.append(conv1_2)
        pool1 = MaxPooling(conv1_2, [2, 2])

        content_cell_1 = BasicConvLSTMCell([self.image_size[0] / 2, self.image_size[1] / 2],
                                           [3, 3], self.gf_dim * 2)
        if not reuse:
            self.content_cell_state_1 = tf.zeros([self.batch_size, self.image_size[0] / 2,
                                                  self.image_size[1] / 2, self.gf_dim * 4])
        h_cell_1, self.content_cell_state_1 = content_cell_1(
            pool1, self.content_cell_state_1, scope="Content_LSTM_1", reuse=reuse)

        conv2_1 = relu(batch_norm(dilated_conv2d(h_cell_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                                                 dilation_rate=1, name='cont_dil_conv2_1', reuse=reuse), name='Content_Conv2_1_Batchnorm', reuse=reuse))
        conv2_2 = relu(batch_norm(dilated_conv2d(conv2_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                                                 dilation_rate=2, name='cont_dil_conv2_2', reuse=reuse), name='Content_Conv2_2_Batchnorm', reuse=reuse))
        res_in.append(conv2_2)
        pool2 = MaxPooling(conv2_2, [2, 2])

        conv3_1 = relu(batch_norm(dilated_conv2d(pool2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                                                 dilation_rate=1, name='cont_dil_conv3_1', reuse=reuse), name='Content_Conv3_1_Batchnorm', reuse=reuse))
        conv3_2 = relu(batch_norm(dilated_conv2d(conv3_1, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                                                 dilation_rate=2, name='cont_dil_conv3_2', reuse=reuse), name='Content_Conv3_2_Batchnorm', reuse=reuse))
        conv3_3 = relu(batch_norm(dilated_conv2d(conv3_2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                                                 dilation_rate=4, name='cont_dil_conv3_3', reuse=reuse), name='Content_Conv3_3_Batchnorm', reuse=reuse))
        res_in.append(conv3_3)
        pool3 = MaxPooling(conv3_3, [2, 2])
        return pool3, res_in

    def comb_layers(self, h_dyn, h_cont, reuse=False):
        comb1 = relu(batch_norm(conv2d(tf.concat(axis=3, values=[h_dyn, h_cont]),
                                       output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                                       d_h=1, d_w=1, name='comb1', reuse=reuse), name='Comb_Conv1_Batchnorm', reuse=reuse))
        comb2 = relu(batch_norm(conv2d(comb1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                                       d_h=1, d_w=1, name='comb2', reuse=reuse), name='Comb_Conv2_Batchnorm', reuse=reuse))
        h_comb = relu(batch_norm(conv2d(comb2, output_dim=self.gf_dim * 4, k_h=3, k_w=3,
                                        d_h=1, d_w=1, name='h_comb', reuse=reuse), name='Comb_H_Batchnorm', reuse=reuse))
        return h_comb

    def residual(self, input_dyn, input_cont, reuse=False):
        n_layers = len(input_dyn)
        res_out = []
        for l in range(n_layers):
            input_ = tf.concat(axis=3, values=[input_dyn[l], input_cont[l]])
            out_dim = input_cont[l].get_shape()[3]
            res1 = relu(batch_norm(conv2d(input_, output_dim=out_dim,
                                          k_h=3, k_w=3, d_h=1, d_w=1,
                                          name='res' + str(l) + '_1', reuse=reuse), name='Residual1_Batchnorm_' + str(l), reuse=reuse))
            res2 = batch_norm(conv2d(res1, output_dim=out_dim, k_h=3, k_w=3,
                                     d_h=1, d_w=1, name='res' + str(l) + '_2', reuse=reuse), name='Residual2_Batchnorm_' + str(l), reuse=reuse)
            res_out.append(res2)
        return res_out

    def dec_cnn(self, h_comb, res_connect, reuse=False):
        shapel3 = [self.batch_size, self.image_size[0] / 4,
                   self.image_size[1] / 4, self.gf_dim * 4]
        shapeout3 = [self.batch_size, self.image_size[0] / 4,
                     self.image_size[1] / 4, self.gf_dim * 2]
        depool3 = FixedUnPooling(h_comb, [2, 2])
        deconv3_3 = relu(batch_norm(deconv2d(relu(tf.add(depool3, res_connect[2])),
                                             output_shape=shapel3, k_h=3, k_w=3,
                                             d_h=1, d_w=1, name='dec_deconv3_3', reuse=reuse), name='Deconv3_3_Batchnorm', reuse=reuse))
        deconv3_2 = relu(batch_norm(deconv2d(deconv3_3, output_shape=shapel3, k_h=3, k_w=3,
                                             d_h=1, d_w=1, name='dec_deconv3_2', reuse=reuse), name='Deconv3_2_Batchnorm', reuse=reuse))
        deconv3_1 = relu(batch_norm(deconv2d(deconv3_2, output_shape=shapeout3, k_h=3, k_w=3,
                                             d_h=1, d_w=1, name='dec_deconv3_1', reuse=reuse), name='Deconv3_1_Batchnorm', reuse=reuse))

        shapel2 = [self.batch_size, self.image_size[0] / 2,
                   self.image_size[1] / 2, self.gf_dim * 2]
        shapeout3 = [self.batch_size, self.image_size[0] / 2,
                     self.image_size[1] / 2, self.gf_dim]
        depool2 = FixedUnPooling(deconv3_1, [2, 2])
        deconv2_2 = relu(batch_norm(deconv2d(relu(tf.add(depool2, res_connect[1])),
                                             output_shape=shapel2, k_h=3, k_w=3,
                                             d_h=1, d_w=1, name='dec_deconv2_2', reuse=reuse), name='Deconv2_2_Batchnorm', reuse=reuse))
        deconv2_1 = relu(batch_norm(deconv2d(deconv2_2, output_shape=shapeout3, k_h=3, k_w=3,
                                             d_h=1, d_w=1, name='dec_deconv2_1', reuse=reuse), name='Deconv2_1_Batchnorm', reuse=reuse))

        shapel1 = [self.batch_size, self.image_size[0],
                   self.image_size[1], self.gf_dim]
        shapeout1 = [self.batch_size, self.image_size[0],
                     self.image_size[1], self.c_dim * 2]
        depool1 = FixedUnPooling(deconv2_1, [2, 2])
        deconv1_2 = relu(batch_norm(deconv2d(relu(tf.add(depool1, res_connect[0])),
                                             output_shape=shapel1, k_h=3, k_w=3, d_h=1, d_w=1,
                                             name='dec_deconv1_2', reuse=reuse), name='Deconv1_2_Batchnorm', reuse=reuse))
        xtp1 = tanh(deconv2d(deconv1_2, output_shape=shapeout1, k_h=3, k_w=3,
                             d_h=1, d_w=1, name='dec_deconv1_1', reuse=reuse))
        return xtp1

    def discriminator(self, image):
        h0 = relu(x=tf_layers.layer_norm(
            conv2d(image, self.df_dim, name='dis_h0_conv')))
        h1 = relu(x=tf_layers.layer_norm(
            conv2d(h0, self.df_dim * 2, name='dis_h1_conv')))
        h2 = relu(x=tf_layers.layer_norm(
            conv2d(h1, self.df_dim * 4, name='dis_h2_conv')))
        h3 = relu(x=tf_layers.layer_norm(
            conv2d(h2, self.df_dim * 8, name='dis_h3_conv')))
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
        if ckpt is None and False:
            self.saver.restore(
                sess, '/lhome/phlippe/scripts/MCNet/models/paper_models/KTH/MCNET.model-98502')
            return True
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            if model_name is None:
                model_name = ckpt_name
            self.saver.restore(sess, os.path.join(checkpoint_dir, model_name))
            return True
        else:
            return False

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
        data_to_mask = (data_to_mask + 1) / 2
        data_to_mask = tf.multiply(data_to_mask, occlusion_mask)
        data_to_mask = (data_to_mask * 2) - 1
        return data_to_mask

    def weighted_BCE_loss(self, predictions, labels, weight0=1, weight1=1):
        with tf.variable_scope("WCE_Loss"):
            predictions = (predictions + 1) / 2.0
            labels = (labels + 1) / 2.0
            epsilon = tf.constant(1e-5, dtype=tf.float32, name="epsilon")
            coef0 = labels * weight0
            coef1 = (1 - labels) * weight1
            coefficient = coef0 + coef1
            label_shape = labels.get_shape().as_list()
            coefficient = tf.ones(label_shape)
            self.max_pred = tf.reduce_max(predictions)
            self.min_pred = tf.reduce_min(predictions)
            self.max_labels = tf.reduce_max(labels)
            self.min_labels = tf.reduce_min(labels)
            print label_shape
            cross_entropy = - 1.0 / label_shape[0] / label_shape[3] * 2 * \
                tf.reduce_sum(tf.multiply(
                    labels * tf.log(predictions + epsilon) +
                    (1 - labels) * tf.log(1 - predictions + epsilon), coefficient))
            cross_entropy_mean = tf.reduce_mean(
                cross_entropy, name="cross_entropy")

            return cross_entropy_mean
