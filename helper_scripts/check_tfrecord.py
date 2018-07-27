import numpy as np
import tensorflow as tf
from glob import glob
import numpy
import os
from argparse import ArgumentParser
import time
CUDA_VISIBLE_DEVICES = []

def main(tfrecord, tfloc, K=9, T=10, seq_steps=1):
    #tfrecord = 'test_0_imgsze=96_seqlen=1_K=9_T=10_size=10'
    #tfloc = 'F:/selfdriving-data/carla_tfrecords/'
    #K = 9
    #T = 10
    #seq_steps = 1

    imgsze_tf, seqlen_tf, K_tf, T_tf, fc_tf, nr_samples, datasze_tf = parse_tfrecord_name(tfrecord)

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



    dataset = tf.data.TFRecordDataset([tfloc+tfrecord + '.tfrecord'])
    dataset = dataset.map(_parse_function)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))

    iterator = dataset.make_initializable_iterator()
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        seq_batch, input_batch, maps_batch, tf_batch = iterator.get_next()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(iterator.initializer)
        #while True:
        s, i, m, t = sess.run([seq_batch, input_batch, maps_batch, tf_batch])
        #print(np.mean(s[0, :, :,:, 0:1]))
        return s, i, m, t

def main_all_data(tfrecord, tfloc, K=9, T=10, seq_steps=1):
    #tfrecord = 'test_0_imgsze=96_seqlen=1_K=9_T=10_size=10'
    #tfloc = 'F:/selfdriving-data/carla_tfrecords/'
    #K = 9
    #T = 10
    #seq_steps = 1

    imgsze_tf, seqlen_tf, K_tf, T_tf, fc_tf, nr_samples, datasze_tf = parse_tfrecord_name(tfrecord)

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

        if (K+T)*seq_steps < input_seq.shape[2]:
            target_seq = target_seq[:,:,:(K+T)*seq_steps,:]
            input_seq = input_seq[:,:,:(K+T)*seq_steps,:]
            maps = maps[:,:,:(K+T)*seq_steps+1,:]
            tf_matrix = tf_matrix[:(K+T)*seq_steps,:,:]

        rgb_cam = rgb_cam[:(K+T)*seq_steps,:,:,:]
        seg_cam = seg_cam[:(K+T)*seq_steps,:,:,:]
        dep_cam = dep_cam[:(K+T)*seq_steps,:,:]
        direction = direction[:(K+T)*seq_steps,:]

        return target_seq, input_seq, maps, tf_matrix, rgb_cam, seg_cam, dep_cam, direction



    dataset = tf.data.TFRecordDataset([tfloc+tfrecord + '.tfrecord'])
    dataset = dataset.map(_parse_function)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(1))

    iterator = dataset.make_initializable_iterator()
    with tf.variable_scope(tf.get_variable_scope()) as vscope:
        seq_batch, input_batch, maps_batch, tf_batch, rgb_cam, seg_cam, dep_cam, direction = iterator.get_next()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        sess.run(iterator.initializer)
        #while True:
        s, i, m, t, rgb, seg, dep, dir = sess.run([seq_batch, input_batch, maps_batch, tf_batch, rgb_cam, seg_cam, dep_cam, direction])
        #print(np.mean(s[0, :, :,:, 0:1]))
        return s, i, m, t, rgb, seg, dep, dir

def parse_tfrecord_name(tfrecordname):
    values = tfrecordname.split('_')
    #print(values)
    dict = {}
    # start at second index, first element contains name
    for var in values:
        try:
            lhs, rhs = var.split('=')
        except:
            continue
        dict[lhs] = rhs
    try:
        w, h = dict['datasze'].split('x')
        return int(dict['imgsze']), int(dict['seqlen']), int(dict['K']), int(dict['T']), int(dict['fc']), int(dict['size']), [int(w), int(h)]
    except:
        try:
            return int(dict['imgsze']), int(dict['seqlen']), int(dict['K']), int(dict['T']), int(dict['fc']), int(dict['size']), [0]
        except:
            return int(dict['imgsze']), int(dict['seqlen']), int(dict['K']), int(dict['T']), int(dict['fc']), 0, [0]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--tfrecord", type=str, dest="tfrecord",
                            required=True, help="Name of the TFrecord to read")
    parser.add_argument("--tfloc", type=str, dest="tfloc",
                            required=True, help="Path to directory where the TFrecord is saved saved")
    parser.add_argument("--K", type=int, dest="K",
                            default=9, help="Value given by TFrecord (in name)")
    parser.add_argument("--T", type=int, dest="T",
                            default=10, help="Value given by TFrecord (in name)")
    parser.add_argument("--seq_steps", type=int, dest="seq_steps",
                            default=1, help="Value needs to satisfy seq_steps*(K+T)<=seqlen which is given in name")

    args = parser.parse_args()
    main(**vars(args))
