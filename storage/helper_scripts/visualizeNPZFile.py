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
    seq = (seq + 1) // 2
    print(seq.shape)
    if motion_content:
        extract_motion_content(seq, out_path, frame_count)
    else:
        extract_occup(seq, out_path, frame_count)

def extract_motion_content(seq, out_path, frame_count):
    seq = np.stack(np.array_split(seq, frame_count, axis=2), axis=2)
    content = seq[:,:,:,::2]
    motion = seq[:,:,:,1:]
    content = np.concatenate([content, np.zeros([128,128,76,1])], axis=3)
    motion = np.concatenate([motion, np.zeros([128,128,76,1])], axis=3)
    for i in range(frame_count):                                               
        scipy.misc.toimage(motion[:,:,i]).save(os.path.join(out_path,'test_motion_'+str(i).zfill(3)+'.png'))
        scipy.misc.toimage(content[:,:,i]).save(os.path.join(out_path,'test_content_'+str(i).zfill(3)+'.png'))

def extract_occup(seq, out_path, frame_count):
    motion_orientation_map = seq[:,:,0:3]
    motion_intensity_map = seq[:,:,3]
    print(f"shape motion {motion_orientation_map.shape} and shape intensity {motion_intensity_map.shape}")
    scipy.misc.toimage(motion_orientation_map).save(os.path.join(out_path,'test_motion_orient.png'))
    scipy.misc.toimage(motion_intensity_map).save(os.path.join(out_path,'test_motion_intens.png'))
    seq = seq[:,:,4:]
    tmp_seq = np.array_split(seq, frame_count, axis=2)
    
    #for i in tmp_seq:
    #    print(i.shape)
    seq = np.stack(tmp_seq[:-5], axis=2) #last couple have 4 instead of 5 values in 3. dim
    #seq = np.concatenate([seq, np.zeros([seq.shape[0],seq.shape[1],seq.shape[2],1])], axis=3)
    for i in range(1): #range(frame_count-5): #occup_map, occlusion_map, lines_map, road_map, combined_mask_map
        scipy.misc.toimage(seq[:,:,i][:,:,0]).save(os.path.join(out_path,'test_occup_'+str(i).zfill(3)+'.png'))
        scipy.misc.toimage(seq[:,:,i][:,:,1]).save(os.path.join(out_path,'test_occlusion_map_'+str(i).zfill(3)+'.png'))
        scipy.misc.toimage(seq[:,:,i][:,:,2]).save(os.path.join(out_path,'test_lines_map_'+str(i).zfill(3)+'.png'))
        scipy.misc.toimage(seq[:,:,i][:,:,3]).save(os.path.join(out_path,'test_road_map_'+str(i).zfill(3)+'.png'))
        scipy.misc.toimage(seq[:,:,i][:,:,4]).save(os.path.join(out_path,'test_combined_mask_map_'+str(i).zfill(3)+'.png'))


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
                            default=100, help="Path to directory where the masks should be saved")
    parser.add_argument("--mc", type=str2bool, dest="motion_content",
                            default=False, help="Path to directory where the masks should be saved")

    args = parser.parse_args()
    main(**vars(args))