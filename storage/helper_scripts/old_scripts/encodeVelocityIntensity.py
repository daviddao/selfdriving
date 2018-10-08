import numpy as np
import scipy.misc
from scipy.ndimage import imread
from glob import glob 
import numpy
import os
from argparse import ArgumentParser
import time

def main(in_path, out_path, thresh, occlusion_path = None, semantic_path = None, step_size = -1):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    vel_imgs = sorted(glob(os.path.join(in_path,"*_velocity.ppm")))
    visited_map = np.zeros([256,256], dtype=int)
    overall_map = np.zeros([256,256], dtype=float)
    step_visited_map = np.zeros([256,256],dtype=int)
    step_overall_map = np.zeros([256,256],dtype=float)
    max_vel = 0.0
    counter = 0
    STEP_FOLDER = 'steps_'+str(step_size).zfill(5)
    if step_size != -1 and not os.path.exists(STEP_FOLDER):
        os.makedirs(STEP_FOLDER)
    start_time = time.time()
    for img_path in vel_imgs:
        counter += 1
        if (time.time() - start_time) > 1:
            print "Processed "+str(counter)+" images, next = "+img_path
            start_time = time.time()
        img_name = img_path.rsplit('/',2)[-1]
        save_path = os.path.join(out_path, img_name)
        
        img = np.array(imread(img_path)).astype(np.float32) 
        vel_intensity = np.hypot(np.absolute(img[:,:,0]-127.5), np.absolute(img[:,:,1]-127.5))
        scipy.misc.toimage(vel_intensity, cmin=0, cmax=180).save(save_path)
        motion_mask = (vel_intensity > thresh)
        
        if semantic_path is not None:
            sem_file = os.path.join(semantic_path, img_name.split(".",-1)[0].rsplit('_',1)[0] + "_semantics.ppm")
            if os.path.isfile(sem_file):
                sem = np.array(imread(sem_file)).astype(np.float32)
                car_sem = (sem[:,:,2] >= 128) * (sem[:,:,1] <= 60) * (sem[:,:,0] <= 60)
                stopping_cars = car_sem * (np.logical_not(motion_mask))
                #stopping_cars = stopping_cars * np.roll(stopping_cars,1,axis=0) * np.roll(stopping_cars,1,axis=1) #* np.roll(stopping_cars,-1,axis=0) * np.roll(stopping_cars,-1,axis=1)
                """
                if np.sum(stopping_cars) > 0:
                    print "Found " + str(np.sum(stopping_cars)) + " stopping cars in semantics: " + sem_file 
                """
                visited_map = visited_map + stopping_cars
                step_visited_map = step_visited_map + stopping_cars
            else:
                print "No semantic file found for "+sem_file
                break


        visited_map = visited_map + motion_mask
        overall_map = overall_map + vel_intensity * motion_mask
        step_visited_map = step_visited_map + motion_mask
        step_overall_map = step_overall_map + vel_intensity * motion_mask
        max_vel = max(np.max(vel_intensity), max_vel)
        
        if step_size > 0 and counter % step_size == 0:
            step_overall_map = visualize_velocity_map(step_overall_map, step_visited_map, max_vel, 0)
            scipy.misc.toimage(step_overall_map, cmin=0, cmax=255).save(os.path.join(STEP_FOLDER,'overall_map_'+str(counter).zfill(7)+'.png'))
            # scipy.misc.toimage((step_visited_map > 0) * 255.0, cmin=0, cmax=255).save(os.path.join(STEP_FOLDER,'visited_map_'+str(counter).zfill(7)+'.png'))
            step_visited_map = np.zeros([256,256],dtype=int)
            step_overall_map = np.zeros([256,256],dtype=float)


    one_channel_map = np.multiply(overall_map, 255.0 / (visited_map + 1e-5) / max_vel) * (visited_map > 20)
    overall_map = visualize_velocity_map(overall_map, visited_map, max_vel)
    scipy.misc.toimage(overall_map, cmin=0, cmax=255).save('overall_map.png')
    scipy.misc.toimage(one_channel_map, cmin=0, cmax=255).save('one_channel_map.png')
    scipy.misc.toimage((visited_map > 0) * 255.0, cmin=0, cmax=255).save('visited_map.png')


def visualize_velocity_map(overall_map, visited_map, max_vel, VISITED_THRESHOLD=20):
    horizon_map = np.array(imread('/lhome/phlippe/dataset/TwoHourSequence_crop/Horizon_map/gridmap_horizon_map.ppm')).astype(np.float32)
    overall_map = np.multiply(overall_map, 255.0 / (visited_map + 1e-5) / max_vel)
    overall_map = np.stack([np.ones([256,256]) * 255, (overall_map - 255.0) * -1, (overall_map - 255.0) * -1], axis=2) * np.stack([(visited_map > VISITED_THRESHOLD)]*3, axis=2)
    overall_map = overall_map * np.stack([(visited_map > VISITED_THRESHOLD)]*3, axis=2) + horizon_map * np.stack([(visited_map <= VISITED_THRESHOLD)]*3, axis=2) * 0.4
    return overall_map  

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, dest="in_path",
                            required=True, help="Path to directory of velocity images")
    parser.add_argument("--outpath", type=str, dest="out_path",
                            required=True, help="Path to directory where the masks should be saved")
    parser.add_argument("--thresh", type=int, dest="thresh",
                            default=10, help="Threshold for velocity map (between 0 and 255*3)")
    parser.add_argument("--occlpath", type=str, dest="occlusion_path",
                            default=None, help="Path to occlusion maps which should be used for blending out. None means no masking")
    parser.add_argument("--sempath", type=str, dest="semantic_path",
                            default=None, help="Path to semantic maps which should be used to detect stopping vehicles. None means no usage of semantic maps")
    parser.add_argument("--step", type=int, dest="step_size",
                            default=-1, help="Saving steps in between of overall map (resets after each save)")
    """
    parser.add_argument("--horizonmap", type=str, dest="horizon_map_path",
                            default=None, help="Path to the horizon map, which is be used as background ")
    """
    args = parser.parse_args()
    main(**vars(args))