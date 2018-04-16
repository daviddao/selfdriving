# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os
from utils import *
import tensorflow as tf

#HEIGHT = 32
#WIDTH = 32
#DEPTH = 3


class Cifar10DataSet(object):

    def __init__(self, data_dir, subset='train', K=9, T=10, sequence_steps=4, tfrecordname_train="", tfrecordname_eval=""):
        self.tfrecordname_train = tfrecordname_train
        self.tfrecordname_eval = tfrecordname_eval
        self.data_dir = data_dir
        self.subset = subset
        self.K = K
        self.T = T
        self.sequence_steps = sequence_steps

    def get_filenames(self):
        if self.subset in ['train', 'validation']:
            self.imgsze_tf, self.seqlen_tf, self.K_tf, self.T_tf = parse_tfrecord_name(self.tfrecordname_train)
            #return [os.path.join(self.data_dir, self.subset + '.tfrecords')] #no .tfrecord since might be folder with sharded records
            return os.path.join(self.data_dir, self.tfrecordname_train) #this might need to be changed, currently always loading same data for train/val/eval
        elif self.subset in ['eval']:
            self.imgsze_tf, self.seqlen_tf, self.K_tf, self.T_tf = parse_tfrecord_name(self.tfrecordname_eval)
            return os.path.join(self.data_dir, self.tfrecordname_eval)
        else:
            raise ValueError('Invalid data subset "%s"' % self.subset)
            

    def parser(self, serialized_example):
        keys_to_features = {'input_seq': tf.FixedLenFeature((), tf.string),
                            'target_seq': tf.FixedLenFeature((), tf.string),
                            'maps': tf.FixedLenFeature((), tf.string),
                            'tf_matrix': tf.FixedLenFeature((), tf.string)}

        parsed_features = tf.parse_single_example(serialized_example, keys_to_features)

        input_batch_shape = [self.imgsze_tf, self.imgsze_tf, self.seqlen_tf*(self.K_tf+self.T_tf), 2]
        seq_batch_shape = [self.imgsze_tf, self.imgsze_tf, self.seqlen_tf*(self.K_tf+self.T_tf), 2]
        maps_batch_shape = [self.imgsze_tf, self.imgsze_tf, self.seqlen_tf*(self.K_tf+self.T_tf)+1, 2]
        transformation_batch_shape = [self.seqlen_tf*(self.K_tf+self.T_tf),3,8]

        input_seq = tf.reshape(tf.decode_raw(parsed_features['input_seq'], tf.float32), input_batch_shape, name='reshape_input_seq')
        target_seq = tf.reshape(tf.decode_raw(parsed_features['target_seq'], tf.float32), seq_batch_shape, name='reshape_target_seq')
        maps = tf.reshape(tf.decode_raw(parsed_features['maps'], tf.float32), maps_batch_shape, name='reshape_maps')
        tf_matrix = tf.reshape(tf.decode_raw(parsed_features['tf_matrix'], tf.float32), transformation_batch_shape, name='reshape_tf_matrix')

        if (self.K+self.T)*self.sequence_steps < input_seq.shape[2]:
            target_seq = target_seq[:,:,:(self.K+self.T)*self.sequence_steps,:]
            input_seq = input_seq[:,:,:(self.K+self.T)*self.sequence_steps,:]
            maps = maps[:,:,:(self.K+self.T)*self.sequence_steps+1,:]
            tf_matrix = tf_matrix[:(self.K+self.T)*self.sequence_steps,:,:]

        return target_seq, input_seq, maps, tf_matrix

    def make_batch(self, batch_size):
        """Read the images and labels from 'filenames'."""
        filenames = self.get_filenames()

        if os.path.isdir(filenames):
            num_records = len(os.listdir(filenames))
            print("Loading from directory. " + str(num_records) + " tfRecords found.")
            files = tf.data.Dataset.list_files(filenames + "/" + "*.tfrecord").shuffle(num_records)
            dataset = files.apply(
                tf.contrib.data.parallel_interleave(
                    lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=256, buffer_size=8*1024*1024),
                    cycle_length=32, sloppy=True)
            )
        else:
            print("Loading from single tfRecord...")
            dataset = tf.data.TFRecordDataset(filenames + ".tfrecord").repeat()
            
        dataset = dataset.map(self.parser, num_parallel_calls=128)
        
        if self.subset == 'train':
            min_queue_examples = int(
                Cifar10DataSet.num_examples_per_epoch(self.subset) * 0.4)
            # Ensure that the capacity is sufficiently large to provide good random
            # shuffling.
            dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)
            
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        dataset = dataset.prefetch(10)
        
        iterator = dataset.make_one_shot_iterator()
        seq_batch, input_batch, map_batch, transformation_batch = iterator.get_next()

        return seq_batch, input_batch, map_batch, transformation_batch
    

    @staticmethod
    def num_examples_per_epoch(subset='train'):
        if subset == 'train':
            return 6000
        elif subset == 'validation':
            return 800
        elif subset == 'eval':
            return 2000
        else:
            raise ValueError('Invalid data subset "%s"' % subset)