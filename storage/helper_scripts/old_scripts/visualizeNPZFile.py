import numpy as np
import scipy.misc
from scipy.ndimage import imread
from glob import glob 
import numpy
import os
from argparse import ArgumentParser
import time

def main(in_path, out_path, frame_count, motion_content):
    seq = np.load(in_path)['arr_0']
    seq = (seq + 1) / 2
    print seq.shape
    if motion_content:
        extract_motion_content(seq, out_path, frame_count)
    else:
        extract_occup(seq, out_path, frame_count)

def extract_motion_content(seq, out_path, frame_count):
    seq = np.stack(np.split(seq, frame_count, axis=2), axis=2)
    content = seq[:,:,:,::2]
    motion = seq[:,:,:,1:]
    content = np.concatenate([content, np.zeros([64,64,81,1])], axis=3)
    motion = np.concatenate([motion, np.zeros([64,64,81,1])], axis=3)
    for i in xrange(frame_count):                                               
        scipy.misc.toimage(motion[:,:,i]).save(os.path.join(out_path,'test_motion_'+str(i).zfill(3)+'.png'))
        scipy.misc.toimage(content[:,:,i]).save(os.path.join(out_path,'test_content_'+str(i).zfill(3)+'.png'))

def extract_occup(seq, out_path, frame_count):
    motion_orientation_map = seq[:,:,0:3]
    motion_intensity_map = seq[:,:,3]
    scipy.misc.toimage(motion_orientation_map).save(os.path.join(out_path,'test_motion_orient.png'))
    scipy.misc.toimage(motion_intensity_map).save(os.path.join(out_path,'test_motion_intens.png'))
    seq = seq[:,:,4:]
    seq = np.stack(np.split(seq, frame_count, axis=2), axis=2)
    seq = np.concatenate([seq, np.zeros([seq.shape[0],seq.shape[1],seq.shape[2],1])], axis=3)
    for i in xrange(frame_count):
        scipy.misc.toimage(seq[:,:,i]).save(os.path.join(out_path,'test_occup_'+str(i).zfill(3)+'.png'))


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, dest="in_path",
                            required=True, help="Path to directory of velocity images")
    parser.add_argument("--outpath", type=str, dest="out_path",
                            required=True, help="Path to directory where the masks should be saved")
    parser.add_argument("--fc", type=int, dest="frame_count",
                            required=True, help="Path to directory where the masks should be saved")
    parser.add_argument("--mc", type=str2bool, dest="motion_content",
                            default=True, help="Path to directory where the masks should be saved")

    args = parser.parse_args()
    main(**vars(args))