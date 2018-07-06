import numpy as np
import argparse
import os
import sys
import tensorflow as tf
from tqdm import tqdm

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) #probably mistake here [value] doesn't work
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main(file_name, data_path, dest_path, all_in_one=True, image_size=96, seq_steps=4, K=9, T=10, useCombinedMask=False, split=1.0):
    """

    Args:
    file_name - Name of .txt where .npz files are listed
    data_path - Destination Location, where file_name file is located
    dest_path - Destination Location, where tfrecords folder is located
    all_in_one - If all .npz files should be written into one .tfrecord or separate records
    record_name - Name of the resulting .TFRecord (deprecated)
    image_size - Size of images in this file
    seq_steps - Length of sequence
    useCombinedMask - determines whether the combined mask of road and occlusion map or only occlusion should be used
    Returns:

    """
    #data_path = '/mnt/ds3lab/daod/mercedes_benz/phlippe/dataset/BigLoop/'
    #file_name = "train_onmove_long_96x96.txt"
    train_list = data_path + file_name
    f = open(train_list, "r")
    f = f.read()
    trainfiles = f.split('\n', -1)
    num_samples = len(trainfiles)-1
    train_files = np.array(trainfiles)
    print(str(num_samples) + " samples")
    info_name = "imgsze=" + str(image_size) + "_seqlen=" + str(seq_steps) + "_K=" + str(K) + "_T=" + str(T) + "_size=" + str(num_samples) + "_"
    if split==1.0:
        converter(train_files, data_path, dest_path, all_in_one, image_size, seq_steps, K, T, useCombinedMask, np.arange(num_samples), info_name+'all')
    else:
        # shuffle data, then split
        shuffler = np.arange(num_samples)
        np.random.shuffle(shuffler)
        split_ind = int(num_samples*split)
        train_samples = shuffler[:split_ind]
        val_samples = shuffler[split_ind:]
        converter(train_files, data_path, dest_path, all_in_one, image_size, seq_steps, K, T, useCombinedMask, train_samples, info_name+'train')
        converter(train_files, data_path, dest_path, all_in_one, image_size, seq_steps, K, T, useCombinedMask, val_samples, info_name+'val')
        
    
    
def converter(train_files, data_path, dest_path, all_in_one, image_size, seq_steps, K, T, useCombinedMask, samples, strname):
    shapes = np.repeat(np.array([image_size]), 1, axis=0)
    sequence_steps = np.repeat(np.array([1 + seq_steps * (K + T)]), 1, axis=0)
    combLoss = np.repeat(useCombinedMask, 1, axis=0)
    if not all_in_one:
        #print('Writing', filename)
        if not os.path.exists(dest_path + 'tfrecords'):
            os.makedirs(dest_path + 'tfrecords')
        for index in tqdm(samples):
            #filename = os.path.join(data_path, record_name + str(index) + '.tfrecord')
            base = os.path.basename(train_files[index])
            base = os.path.splitext(base)[0]
            filename = os.path.join(dest_path + 'tfrecords', base + '_' + strname + '.tfrecord')
            #print("writing "+base)
            tfiles = train_files[index]
            with tf.python_io.TFRecordWriter(filename) as writer:
                for f, img_sze, seq, useCM in zip([tfiles], shapes, sequence_steps, combLoss):
                    target_seq, input_seq, maps, tf_matrix = load_gridmap_onmove_tfrecord(f, img_sze, seq, useCM)
                #target_seq, input_seq, maps, tf_matrix = load_gridmap_onmove_tfrecord(train_files[index], image_size, seq_steps, useCombinedMask)
                input_batch_shape = input_seq.shape
                seq_batch_shape = target_seq.shape
                maps_batch_shape = maps.shape
                transformation_batch_shape = tf_matrix.shape
                seq_batch = target_seq.tostring()
                input_batch = input_seq.tostring()
                map_batch = maps.tostring()
                transformation_batch = tf_matrix.tostring()
                #image_raw = images[index].tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            #'input_batch_shape': _int64_feature(input_batch_shape),
                            #'seq_batch_shape': _int64_feature(seq_batch_shape),
                            #'maps_batch_shape': _int64_feature(maps_batch_shape),
                            #'transformation_batch_shape': _int64_feature(transformation_batch_shape),
                            'input_seq': _bytes_feature(input_batch),
                            'target_seq': _bytes_feature(seq_batch),
                            'maps': _bytes_feature(map_batch),
                            'tf_matrix': _bytes_feature(transformation_batch)
                        }))
                writer.write(example.SerializeToString())
                
    else:
        base = "all_in_one"
        filename = os.path.join(dest_path, base + '_' + strname + '.tfrecord')
        with tf.python_io.TFRecordWriter(filename) as writer:
            #filename = os.path.join(data_path, record_name + str(index) + '.tfrecord')
            #print("writing "+base)
            for index in tqdm(samples):
                tfiles = train_files[index]
                for f, img_sze, seq, useCM in zip([tfiles], shapes,sequence_steps,combLoss):
                    target_seq, input_seq, maps, tf_matrix = load_gridmap_onmove_tfrecord(f, img_sze, seq, useCM)
                input_batch_shape = input_seq.shape
                seq_batch_shape = target_seq.shape
                maps_batch_shape = maps.shape
                transformation_batch_shape = tf_matrix.shape
                seq_batch = target_seq.tostring()
                input_batch = input_seq.tostring()
                map_batch = maps.tostring()
                transformation_batch = tf_matrix.tostring()
                #image_raw = images[index].tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            #'input_batch_shape': _int64_feature(input_batch_shape),
                            #'seq_batch_shape': _int64_feature(seq_batch_shape),
                            #'maps_batch_shape': _int64_feature(maps_batch_shape),
                            #'transformation_batch_shape': _int64_feature(transformation_batch_shape),
                            'input_seq': _bytes_feature(input_batch),
                            'target_seq': _bytes_feature(seq_batch),
                            'maps': _bytes_feature(map_batch),
                            'tf_matrix': _bytes_feature(transformation_batch)
                        }))
                writer.write(example.SerializeToString())
            
        #writer.close()
    print("Finished Writing.")
            
def load_gridmap_onmove_tfrecord(f_name, image_size, frame_count, useCombinedMask=False):
    """
    Load function for gridmaps consisting of occupancy, occlusion, horizon map (road and lines) and combined mask of occlusion and map.
    Works like load_gridmap_motion_maps, but uses instead of the motion maps the horizon map of each time frame. For better understanding
    the map has to be 2 channel (road itself 1|-1 and lines 1|-1). In addition to select whether only objects on the road should be taken into
    the loss function or every object the last channel has to provide a map which combined the occlusion map with the road map.
    So the array must have a shape of [image_size, image_size, frame_count * 5].

    Args:
        f_name - Name of compressed numpy file which should be loaded
        image_size - Size of images in this file
        frame_count - Number of images which should be loaded
        useCombinedMask - determines whether the combined mask of road and occlusion map or only occlusion should be used

    Returns:
        target_seq - sequence of frames in an array of shape [image_size, image_size, frame_count, 2]
        input_seq - sequence of frames for the input. Includes occupancy and occlusion map [image_size, image_size, frame_count + 1, 2]
        maps - sequence of the horizon map. Channel 0 lines, channel 1 road. The shape is [image_size, image_size, frame_count, 2]
    """
    #print(f_name)
    f_name = f_name.split('\n', -1)[0]
    #print(f_name)
    seq = np.load(f_name)['arr_0']
    # Split [image_size, image_size, frame_count * 2] to [image_size, image_size, frame_count, 2]
    seq = np.stack(np.split(seq[:,:,:frame_count*5], frame_count, axis=2), axis=2)
    if image_size < seq.shape[0]:
        orig_size = seq.shape[0]
        start_index = (orig_size - image_size) // 2
        end_index = start_index + image_size
        seq = seq[start_index:end_index,start_index:end_index]

    input_seq = seq[:,:,:-1,0:2]
    maps = seq[:,:,:,2:4]
    if useCombinedMask:
        loss_mask = seq[:,:,1:,4:5]
        input_seq[:,:,:,0:1] = np.multiply((seq[:,:,:-1,4:5] + 1) // 2, (input_seq[:,:,:,0:1] + 1) // 2) * 2 - 1
    else:
        loss_mask = seq[:,:,1:,1:2]
        input_seq[:,:,:,0:1] = np.multiply((seq[:,:,:-1,4:5] + 1) // 2, (input_seq[:,:,:,0:1] + 1) // 2) * 2 - 1
        seq[:,:,1:,0:1] = np.multiply((seq[:,:,1:,0:1] + 1) // 2, (seq[:,:,1:,3:4] + 1) / 2) * 2 - 1
    target_seq = np.concatenate([seq[:,:,1:,0:1],loss_mask], axis=3)

    p = "." if len(f_name.split(".",-1)) > 1 else ""
    tf_name = p + ''.join(f_name.split('.',-1)[:-1]) + "_transformation.npz"
    #print(tf_name)
    try:
        tf_matrix = np.load(tf_name)['arr_0'][:frame_count-1]
        tf_matrix[:,:,2] = tf_matrix[:,:,2]
        tf_matrix[:,:,5] = - tf_matrix[:,:,5]
    except IOError:
        print("IOError for " + tf_name)
        tf_matrix = np.zeros([frame_count-1, 3, 8], dtype=np.float32)
        tf_matrix[:,:,0] = 1
        tf_matrix[:,:,4] = 1

    return target_seq, input_seq, maps, tf_matrix
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--directory',
      type=str,
      default='/mnt/ds3lab/daod/mercedes_benz/phlippe/dataset/BigLoop/',
        dest="data_path",
      help='Directory to write the converted result'
    )
    parser.add_argument("--filename", type=str, dest="file_name",
                        default='train_onmove_long_96x96.txt', help="Name of .txt where .npz files are listed")
    parser.add_argument("--data_path", type=str, dest="data_path",
                        default='/mnt/ds3lab-scratch/lucala/phlippe/dataset/BigLoop/', help="Destination Location, where file is located")
    parser.add_argument("--dest_path", type=str, dest="dest_path",
                        default='/mnt/ds3lab-scratch/lucala/phlippe/dataset/BigLoop/', help="Destination Location, where tfrecords folder is located")
    parser.add_argument("--all_in_one", type=bool, dest="all_in_one",
                        default=True, help="If all .npz files should be written into one .tfrecord or separate records")
    parser.add_argument("--image_size", type=int, dest="image_size",
                        default=96, help="Size of images in this file")
    parser.add_argument("--seq_steps", type=int, dest="seq_steps",
                        default=4, help="Length of sequence")
    parser.add_argument("--useCombinedMask", type=bool, dest="useCombinedMask",
                        default=False, help="determines whether the combined mask of road and occlusion map or only occlusion should be used")
    parser.add_argument("--K", type=int, dest="K",
                        default=9, help="Ground Truth Length")
    parser.add_argument("--T", type=int, dest="T",
                        default=10, help="")
    parser.add_argument("--split", type=float, dest="split",
                        default=1.0, help="Percentage of data used for training")
    args = parser.parse_args()
    main(**vars(args))