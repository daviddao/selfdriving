import numpy as np
import scipy.misc
from scipy.ndimage import imread
from glob import glob 
import numpy
import os
from argparse import ArgumentParser
import time
# from matplotlib.colors import rgb_to_hsv

def main(in_path, out_path, thresh, occlusion_path = None, step_size = -1, image_size = 256):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    vel_imgs = sorted(glob(os.path.join(in_path,"*")))
    overall_img = np.zeros([image_size, image_size], dtype=bool)
    avg_img = np.zeros([image_size,image_size,3], dtype=np.float32)
    step_avg_img = np.zeros([image_size,image_size,3],dtype=np.float32)
    last_images = []
    ped_tracker = np.zeros([image_size,image_size], dtype=int)
    car_tracker = np.zeros([image_size,image_size], dtype=int)
    counter = 0
    STEP_FOLDER = 'orientation_steps_'+str(step_size).zfill(5)
    if step_size != -1 and not os.path.exists(STEP_FOLDER):
        os.makedirs(STEP_FOLDER)
    start_time = time.time()
    for img_path in vel_imgs:
        counter += 1
        """
        if counter < 0:
            continue
        if counter > 5000:
            break
        """
        shouldSaveImage = True
        if (time.time() - start_time) > 1:
            print "Processed "+str(counter)+" images, next = "+img_path
            start_time = time.time()
        img_name = img_path.rsplit('/',2)[-1]
        save_path = os.path.join(out_path, img_name)
        
        img = np.array(imread(img_path)).astype(np.float32)
        avg_img = avg_img + (img > thresh) * img 
        step_avg_img = step_avg_img + (img > thresh) * img 
        last_images.append( (img > thresh) * img )
        single_image = (img > thresh) * img
        img = np.sum(img, axis=2)
        overall_img = overall_img + (img > thresh)
        img = (img > thresh) * 1.0


        semantic_path = os.path.join('/lhome/phlippe/dataset/TwoHourSequence_crop/Sequence2/semantics_128x128/', img_name.split(".",-1)[0].rsplit('_',1)[0]+"_semantics.ppm")
        if os.path.isfile(semantic_path):
            sem = np.array(imread(semantic_path)).astype(np.float32)
            ped_sem = (sem[:,:,0] >= 255) * (sem[:,:,1] <= 0) * (sem[:,:,2] <= 0)
            car_sem = (sem[:,:,2] >= 128) * (sem[:,:,1] <= 60) * (sem[:,:,0] <= 60)
            stopping_cars = car_sem * (np.logical_not(img))
            #stopping_cars = stopping_cars * np.roll(stopping_cars,1,axis=0) * np.roll(stopping_cars,1,axis=1) * np.roll(stopping_cars,-1,axis=0) * np.roll(stopping_cars,-1,axis=1)
            """
            if np.sum(ped_sem) != 0:
                print "Found pedestrian - "+img_path 
            """
            ped_tracker = ped_tracker + ped_sem
            car_tracker = car_tracker + car_sem
            step_avg_img = step_avg_img + np.stack([stopping_cars * 255] * 3, axis=2)
            last_images[-1] = last_images[-1] + np.stack([stopping_cars * 255] * 3, axis=2)
            single_image = single_image + np.stack([stopping_cars * 255] * 3, axis=2)
            img = img + stopping_cars
        else:
            print "Semantic File not found for " + img_path + "(searched for "+semantic_path+")"
            break

        if occlusion_path is not None:
            occl_file = os.path.join(occlusion_path, img_name.split(".",-1)[0].rsplit('_',1)[0] + "_occupancy_occlusion.pgm")
            try:
                occl = imread(occl_file)
                img = np.multiply(img, occl)
                single_image = np.multiply(single_image, np.stack([occl]*3, axis=2))
            except IOError:
                print "File not found: "+occl_file
                shouldSaveImage = False

        if shouldSaveImage:
            scipy.misc.toimage(img).save(save_path)
        else:
            print "File not saved: "+save_path
        """
        if step_size > 0 and counter % step_size == 0:
            step_avg_img = visualize_avg_img(step_avg_img, thresh=1)
            scipy.misc.toimage(step_avg_img, cmin=0, cmax=255).save(os.path.join(STEP_FOLDER, "avg_map_"+str(counter).zfill(7)+".png"))
            step_avg_img = np.zeros([256,256,3],dtype=np.float32)
        """
        if step_size > 0 and counter > step_size:
            temp_img = sum(last_images)
            temp_img = visualize_avg_img(temp_img, thresh=1)
            scipy.misc.toimage(temp_img, cmin=0, cmax=255).save(os.path.join(STEP_FOLDER, "avg_map_"+str(counter).zfill(7)+".png"))
            del last_images[0]
        if step_size <= 0:
            del last_images[0]
            



    
    avg_img = visualize_avg_img(avg_img, horizon = False)
    avg_white = np.sum(avg_img, axis=2)
    avg_white = np.multiply(avg_white > 0.5,np.ones(avg_white.shape) * 255)

    horizon_map = np.array(imread('/lhome/phlippe/dataset/TwoHourSequence_crop/Horizon_map/gridmap_horizon_map_128x128.ppm')).astype(np.float32)
    ped_horizon = np.stack([(ped_tracker > 0) * 255, np.zeros([image_size,image_size]), np.zeros([image_size,image_size])], axis=2) + horizon_map * np.stack([(ped_tracker <= 0)]*3, axis=2) * 0.7
    avg_horizon = np.multiply(avg_img, np.stack([avg_white > 0.5]*3, axis=2)) + np.multiply(horizon_map, np.stack([avg_white <= 0.5]*3, axis=2)) * 0.5


    ped_tracker = np.stack([(ped_tracker > 0) * 255, np.zeros([image_size,image_size]), np.zeros([image_size,image_size])], axis=2) + np.stack([avg_white * (ped_tracker <= 0) * 0.5]*3, axis=2)
    overall_img = overall_img * 1.0

    
    scipy.misc.toimage(avg_img, cmin=0, cmax=255).save('avg.png')
    scipy.misc.toimage(overall_img, cmin=0, cmax=255).save('overall.png')
    scipy.misc.toimage(ped_tracker * 255.0, cmin=0, cmax=255).save('ped.png')
    scipy.misc.toimage(car_tracker * 255.0, cmin=0, cmax=255).save('car.png')
    scipy.misc.toimage(avg_white * 255, cmin=0, cmax=255).save('avg_white.png')
    scipy.misc.toimage(ped_horizon, cmin=0, cmax=255).save('ped_horizon.png')
    scipy.misc.toimage(avg_horizon, cmin=0, cmax=255).save('avg_horizon.png')

def visualize_avg_img(avg_img, thresh=7.5, horizon=True):
    horizon_map = np.array(imread('/lhome/phlippe/dataset/TwoHourSequence_crop/Horizon_map/gridmap_horizon_map_128x128.ppm')).astype(np.float32)
    factor_array = np.tile(255.0 / (np.reshape(np.max(avg_img, axis=2)+1e-10,[avg_img.shape[0],avg_img.shape[1],1])), [1,1,3])
    factor_array = (factor_array <= 1/thresh) * factor_array
    avg_img = np.multiply(avg_img, factor_array)
    avg_white = np.sum(avg_img, axis=2)
    avg_white = np.multiply(avg_white > 0.5,np.ones(avg_white.shape) * 255)
    avg_horizon = np.multiply(avg_img, np.stack([avg_white > 0.5]*3, axis=2)) + np.multiply(horizon_map, np.stack([avg_white <= 0.5]*3, axis=2)) * 0.5
    if horizon:
        return avg_horizon
    else:
        return avg_img

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, dest="in_path",
                            required=True, help="Path to directory of velocity images")
    parser.add_argument("--outpath", type=str, dest="out_path",
                            required=True, help="Path to directory where the masks should be saved")
    parser.add_argument("--thresh", type=int, dest="thresh",
                            default=10, help="Threshold for sum of pixels (between 0 and 255*3)")
    parser.add_argument("--occlpath", type=str, dest="occlusion_path",
                            default=None, help="Path to occlusion maps which should be used for blending out. None means no masking")
    parser.add_argument("--step", type=int, dest="step_size",
                            default=-1, help="Step size to save avg img")
    parser.add_argument("--imsize", type=int, dest="image_size",
                            default=256, help="Size of the images used in here")

    args = parser.parse_args()
    main(**vars(args))