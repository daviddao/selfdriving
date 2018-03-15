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

    def __init__(self, image_size, batch_size=32, c_dim=1,  # Input has to be 2 dimensional: First occupancy, last occlusion map
                 K=10, T=10, checkpoint_dir=None, is_train=True,
                 iterations=1, d_input_frames=20, useSELU=False,
                 motion_map_dims=2, showFutureMaps=True,
                 predOcclValue=-1, useSmallSharpener=False,
                 gpu=[0], useGAN=True):

        self.batch_size = batch_size
        self.image_size = image_size
        self.is_train = is_train
        self.d_input_frames = d_input_frames
        self.useSELU = useSELU
        self.maps_offset = 1 if showFutureMaps else 0
        self.predOcclValue = predOcclValue
        self.small_sharpener = useSmallSharpener
        self.gpu = gpu
        self.useGAN = useGAN

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
        print("Target shape = " + str(self.target_shape))
        print("Ego motion shape = " + str(self.ego_motion_shape))
        print("K = " + str(K) + ", T = " + str(T))
        self.build_model()

    def build_model(self):
        self.input_tensor = tf.placeholder(
            tf.float32, self.input_shape, name='input_tensor')
        self.motion_map_tensor = tf.placeholder(
            tf.float32, self.motion_map_shape, name='motion_map_tensor')
        self.target = tf.placeholder(
            tf.float32, self.target_shape, name='target')
        self.pred_occlusion_map = tf.ones(
            self.occlusion_shape, dtype=tf.float32, name='Pred_Occlusion_Map') * self.predOcclValue
        self.ego_motion = tf.placeholder(
            tf.float32, self.ego_motion_shape, name='ego_motion')

        pred, trans_pred, pre_trans_pred = self.forward(self.input_tensor, self.motion_map_tensor, self.ego_motion)
        self.G = tf.stack(axis=3, values=pred)
        self.G_trans = tf.stack(axis=3, values=trans_pred)
        self.G_before_trans = tf.stack(axis=3, values=pre_trans_pred)

        if self.is_train:
            # with tf.device('/gpu:%d' % (self.gpu[0])):
            self.loss_occlusion_mask = (tf.tile(
                self.target[:, :, :, :, -1:], [1, 1, 1, 1, self.c_dim]) + 1) / 2.0
            self.target_masked = self.mask_black(
                self.target[:, :, :, :, :self.c_dim], self.loss_occlusion_mask)
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
            if len(self.gpu) > 1:
                D_gpu = self.gpu[1]
            else:
                D_gpu = self.gpu[0]
            with tf.device('/gpu:%d' % (D_gpu)):
                if self.useGAN:
                    start_frame = 0
                    center_frame = self.d_input_frames // 2
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
                else:
                    self.d_loss_fake = tf.constant(0.0)
                    self.d_loss_real = tf.constant(0.0)
                    self.d_loss = tf.constant(0.0)
                    self.L_GAN = tf.constant(0.0)

            self.L_p = tf.reduce_mean(
                tf.square(self.G_masked - self.target_masked)
            )
            # self.L_gdl = gdl(gen_sim, true_sim, 1.)
            # self.L_img = self.L_p + self.L_gdl
            self.L_BCE = 0

            if len(self.gpu) > 1:
                loss_gpu = self.gpu[1:]
            else:
                loss_gpu = self.gpu
            for i, gpu_index in enumerate(loss_gpu):
                index_step = (self.iterations * (self.K + self.T) // len(loss_gpu))
                start_index = i * index_step
                end_index = (i + 1) * index_step
                """
                if i == 0:
                    start_index = start_index + self.K
                """
                with tf.device('/gpu:%d' % (gpu_index)):
                    self.L_BCE += self.weighted_BCE_loss(
                        self.G_masked[:,:,:,start_index:end_index,:], self.target_masked[:,:,:,start_index:end_index,:])
            """
            self.L_BCE += self.weighted_BCE_loss(
                    self.G_masked[:,:,:,self.K:,:], self.target_masked[:,:,:,self.K:,:])
            """
            self.L_img = self.L_BCE

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

    def forward(self, input_tensor, motion_maps, ego_motions):

        reuse = False
        pred = []
        trans_pred = []
        pre_trans_pred = []
        for iter_index in range(self.iterations):
            if len(self.gpu) > 1:
                i = iter_index // (self.iterations // (len(self.gpu) - 1)) + 1
                i = max(0, min(i, len(self.gpu) - 1))
                gpu_index = self.gpu[i]
            else:
                gpu_index = self.gpu[0]
            with tf.device('/gpu:%d' % (gpu_index)):
                print("Iteration " + str(iter_index) + " on /gpu:"+str(gpu_index))
                with tf.name_scope('iter_%d' % (iter_index)):
                  # Ground Truth as Input
                    for t in range(self.K):
                        timestep = iter_index * (self.K + self.T) + t
                        motion_enc_input = tf.concat([input_tensor[:, :, :, timestep, :], motion_maps[:,:,:,timestep,:]], axis=3)
                        transform_matrix = ego_motions[:,timestep]
                        h_motion, res_m = self.motion_enc(motion_enc_input, reuse=reuse)
                        decoded_output = self.dec_cnn(h_motion, res_m, reuse=reuse)
                        pre_trans_pred.append(tf.identity(decoded_output))
                        prediction_output = decoded_output # self.motion_maps_combined(motion_maps[:,:,:,timestep + self.maps_offset,:], decoded_output, reuse=reuse)
                        prediction_output = spatial_transformer_network((prediction_output + 1) / 2, transform_matrix[:,0,:6]) * 2 - 1
                        trans_pred.append(tf.identity(prediction_output))
                        # prediction_output = self.sharpen_image(prediction_output, motion_maps[:,:,:,timestep+1,:], motion_enc_input[:,:,:,:2], reuse=reuse)
                        self.transform_hidden_states(transform_matrix)
                        pred.append(prediction_output)
                        reuse = True

                  # Prediction sequence
                    for t in range(self.T):
                        timestep = iter_index * (self.K + self.T) + self.K + t
                        motion_enc_input = tf.concat(
                          [self.keep_alive(pred[-1]), self.pred_occlusion_map, motion_maps[:,:,:,timestep,:]], axis=3)
                        transform_matrix = ego_motions[:,timestep]
                        h_motion, res_m = self.motion_enc(motion_enc_input, reuse=reuse)
                        decoded_output = self.dec_cnn(h_motion, res_m, reuse=reuse)
                        pre_trans_pred.append(tf.identity(decoded_output))
                        prediction_output = decoded_output # self.motion_maps_combined(motion_maps[:,:,:,timestep + self.maps_offset,:], decoded_output, reuse=reuse)
                        prediction_output = spatial_transformer_network((prediction_output + 1) / 2, transform_matrix[:,0,:6]) * 2 - 1
                        trans_pred.append(tf.identity(prediction_output))
                        # prediction_output = self.sharpen_image(prediction_output, motion_maps[:,:,:,timestep+1,:], motion_enc_input[:,:,:,:2], reuse=reuse)
                        self.transform_hidden_states(transform_matrix)
                        pred.append(prediction_output)

        return pred, trans_pred, pre_trans_pred

    def motion_enc(self, motion_in, reuse):
        res_in = []
        conv1_1 = relu(conv2d(motion_in, output_dim=self.gf_dim, k_h=5, k_w=5,
                              d_h=2, d_w=2, name='mot_conv1_1', reuse=reuse), useSELU=self.useSELU)
        res_in.append(conv1_1)

        motion_cell_1 = BasicConvLSTMCell([self.image_size[0] // 2, self.image_size[1] // 2],
                                          [3, 3], self.gf_dim)
        if not reuse:
            self.motion_cell_state_1 = tf.zeros([self.batch_size, self.image_size[0] // 2,
                                                 self.image_size[1] // 2, self.gf_dim * 2])
        h_cell_1, self.motion_cell_state_1 = motion_cell_1(
            conv1_1, self.motion_cell_state_1, scope="Motion_LSTM_1", reuse=reuse)

        conv2_1 = relu(conv2d(h_cell_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                              d_h=2, d_w=2, name='mot_conv2_1', reuse=reuse), useSELU=self.useSELU)

        res_in.append(conv2_1)

        motion_cell_2 = BasicConvLSTMCell([self.image_size[0] // 4, self.image_size[1] // 4],
                                          [3, 3], self.gf_dim * 2)
        if not reuse:
            self.motion_cell_state_2 = tf.zeros([self.batch_size, self.image_size[0] // 4,
                                                 self.image_size[1] // 4, self.gf_dim * 4])
        h_cell_2, self.motion_cell_state_2 = motion_cell_2(
            conv2_1, self.motion_cell_state_2, scope="Motion_LSTM_2", reuse=reuse)

        conv3_1 = relu(dilated_conv2d(h_cell_2, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                                                 dilation_rate=1, name='mot_dil_conv3_1', reuse=reuse), useSELU=self.useSELU)
        conv3_2 = relu(dilated_conv2d(conv3_1, output_dim=self.gf_dim * 2, k_h=3, k_w=3,
                                                 dilation_rate=2, name='mot_dil_conv3_2', reuse=reuse), useSELU=self.useSELU)

        motion_cell_3 = BasicConvLSTMCell([self.image_size[0] // 4, self.image_size[1] // 4],
                                          [3, 3], self.gf_dim * 2)
        if not reuse:
            self.motion_cell_state_3 = tf.zeros([self.batch_size, self.image_size[0] // 4,
                                                 self.image_size[1] // 4, self.gf_dim * 4])
        h_cell_3, self.motion_cell_state_3 = motion_cell_3(
            conv3_2, self.motion_cell_state_3, scope="Motion_LSTM_3", reuse=reuse)

        return h_cell_3, res_in


    def dec_cnn(self, h_comb, res_connect, reuse=False):

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
        if ckpt is None and False:
            self.saver.restore(
                sess, '/lhome/lucala/scripts/MCNet/models/paper_models/KTH/MCNET.model-98502')
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
