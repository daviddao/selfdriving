"""
This file contains operations used for setting up the MCNetwork with deep Tracking in mcnet_deep_tracking.
Python 2

Strongly inspired by https://github.com/bioinf-jku/SNNs/blob/master/selu.py for selu function
GNU General Public License v3.0 - downloaded 07/11/2017

Strongly inspired by https://github.com/rubenvillegas/iclr2017mcnet/blob/master/src/ops.py
MIT License - download 07/01/2017
"""

import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *


def selu(x=None, scope_name="SELU"):
    """
    Scaled ELU activation function proposed by Klambauer et al.: Self-Normalizing Neural Networks (https://arxiv.org/pdf/1706.02515.pdf)
    Args:
      x - Input to SELU
      scope_name - Name scope in which the unit should be defined
    """
    with ops.name_scope(scope_name) as scope:
        # Constants for activation function
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        # SELU activation function
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def dilated_conv2d(input_, output_dim,
                   k_h=3, k_w=3, dilation_rate=1,
                   name="dil_conv2d", reuse=False, padding='SAME'):
    """
    Builds up a dilated convolution layer on input including weights and biases.
    Args:
      input_ - Input to the layer
      output_dim - Number of output channels
      k_h - Kernel height
      k_w - Kernel width
      dilation_rate - Dilation rate (for details see Tensorflow atrous_conv2d)
      name - Name of variable scope
      reuse - If variables should be reused
      padding - Padding of convolution layer
    """
    with tf.variable_scope(name, reuse=reuse):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.contrib.layers.xavier_initializer())
        conv = tf.nn.atrous_conv2d(
            input_, w, rate=dilation_rate, padding=padding)

        biases = tf.get_variable('biases', [output_dim],
                                 initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv
