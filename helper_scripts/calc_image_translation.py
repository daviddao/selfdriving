import cv2
import numpy as np
from glob import glob
from scipy.ndimage import imread
import scipy.misc
from argparse import ArgumentParser
import os
import math


def get_transformation_matrix(imsize, theta, dx, dy):
    center = (imsize - 1) / 2.0
    a11 = math.cos(theta)
    a12 = -math.sin(theta)
    a13 = dx + (math.cos(theta) - math.sin(theta)) * -center + center
    a21 = math.sin(theta)
    a22 = math.cos(theta)
    a23 = dy + (math.sin(theta) + math.cos(theta)) * -center + center
    M = np.array([[a11, a12, a13], [a21, a22, a23]])
    return M


def get_STM_matrix(imsize, theta, dx, dy):
    theta = -theta
    a11 = math.cos(theta)
    a12 = -math.sin(theta)
    a13 = dx / ((imsize - 1) / 2.0)
    a21 = math.sin(theta)
    a22 = math.cos(theta)
    a23 = dy / ((imsize - 1) / 2.0)
    M = np.array([[a11, a12, a13], [a21, a22, a23]])
    return M


def get_transformation_parameter(imsize, gridmap_size, frame_rate, vel, yaw_rate):
    period_duration = 1.0 / frame_rate
    yaw_diff = math.radians(yaw_rate * period_duration)
    pixel_size = gridmap_size * 1.0 / imsize    # [m]
    pixel_diff = vel * period_duration * 1.0 / pixel_size
    pixel_diff_y = math.cos(yaw_diff) * pixel_diff
    pixel_diff_x = math.sin(yaw_diff) * pixel_diff
    return yaw_diff, pixel_diff_x, pixel_diff_y

def get_combined_transformation_parameter(imsize, gridmap_size, frame_rate, vels, yaw_rates):
    period_duration = 1.0 / frame_rate
    pixel_size = gridmap_size * 1.0 / imsize    # [m]
    yaw_diff = 0
    pixel_diff_x = 0
    pixel_diff_y = 0
    for index in xrange(min(len(yaw_rates), len(vels))):
        pre_yaw_diff = yaw_diff
        step_yaw_diff = math.radians(yaw_rates[index] * period_duration)
        yaw_diff = yaw_diff + step_yaw_diff
        pixel_diff = vels[index] * period_duration * 1.0 / pixel_size
        step_radius = pixel_diff / math.fabs(step_yaw_diff)
        step_y_diff = math.sin(math.fabs(step_yaw_diff)) * step_radius
        step_x_diff = (1 - math.cos(math.fabs(step_yaw_diff))) * step_radius * (-1 if step_yaw_diff < 0 else 1)
        alt_diff_y = math.cos(step_yaw_diff) * pixel_diff
        alt_diff_x = math.sin(step_yaw_diff) * pixel_diff
        pixel_diff_x += step_x_diff
        pixel_diff_y += step_y_diff
        pixel_diff_x =  + math.cos(step_yaw_diff) * pixel_diff_x + math.sin(step_yaw_diff) * pixel_diff_y
        pixel_diff_y =  - math.sin(step_yaw_diff) * pixel_diff_x + math.cos(step_yaw_diff) * pixel_diff_y
        if step_y_diff / alt_diff_y > 1.5 or alt_diff_y / step_y_diff > 1.5:
            print "Velocity = "+str(vels[index])
            print "Yaw rate = "+str(yaw_rates[index])
            print "Radius = "+str(step_radius)
            print "Step_y = "+str(step_y_diff)
            print "Step_x = "+str(step_x_diff)
            print "Other step_y = "+str(alt_diff_y)
            print "Other step_x = "+str(alt_diff_x)
        

    #print "Overall pixel_diff_x = "+str(pixel_diff_x)+", pixel_diff_y = "+str(pixel_diff_y)
    return yaw_diff, pixel_diff_x, pixel_diff_y 


def load_odometry_file(file_path):
    dictionary = dict()
    f = open(file_path, "r")
    for line in f:
        line_elements = line.split(" ", -1)
        frame_number = line_elements[0]
        yaw_rate = float(line_elements[4])
        vel = float(line_elements[5])
        dictionary[frame_number] = {'yaw_rate': yaw_rate, 'vel': vel}
    return dictionary


def get_gridmap_size(file_path):
    f = open(file_path, "r")
    comments = f.readline()
    data = f.readline().split(" ", -1)
    #return float(data[0]) * float(data[2]) * float(data[5])
    return 45.6

def get_frame_number(frame_path):
    return frame_path.split("/", -1)[-1].split(".", -1)[0].split("_", -1)[1]


def transform_image(odometry_dict, image, frame_number, gridmap_size, frame_rate):
    theta, dx, dy = get_transformation_parameter(image.shape[0], gridmap_size, frame_rate, odometry_dict[
                                                 frame_number]['vel'], odometry_dict[frame_number]['yaw_rate'])
    transform_matrix = get_transformation_matrix(image.shape[0], theta, dx, dy)
    return cv2.warpAffine(image, transform_matrix, (image.shape[0], image.shape[1]))


def load_image(frame_path):
    img = imread(frame_path)
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=2)
    if img.shape[2] == 1:
        img = np.concatenate([img] * 3, axis=2)
    return img


def main(frame_folder, odometry_file, gridmap_file, out_path, frame_rate=10, combined_frames=1, overwrite=False):
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    odometry_dict = load_odometry_file(odometry_file)
    gridmap_size = get_gridmap_size(gridmap_file)
    frames = sorted(glob(os.path.join(frame_folder, "*")))
    print "Gridmap size = "+str(gridmap_size)
    img = None
    counter = 0

    for frame_index, frame in enumerate(frames):
        img = load_image(frame)
        frame_number = get_frame_number(frame)
        if frame_number in odometry_dict:
            counter += 1
            save_path = os.path.join(
                out_path, "gridmap_" + frame_number + "_translation.npz")
            if overwrite or not os.path.isfile(save_path):
                transform_matrix = np.zeros([3, 2, 3])
                for i in xrange(transform_matrix.shape[0]):
                    if combined_frames == 1 or (len(frames) - frame_index <= combined_frames):
                        theta, dx, dy = get_transformation_parameter(img.shape[0] / (2 ** i), gridmap_size, frame_rate, odometry_dict[
                                                                 frame_number]['vel'], odometry_dict[frame_number]['yaw_rate'])
                    else:
                        vels = []
                        yaw_rates = []
                        for comb_index in xrange(combined_frames):
                            vels.append(odometry_dict[get_frame_number(frames[frame_index + comb_index])]['vel'])
                            yaw_rates.append(odometry_dict[get_frame_number(frames[frame_index + comb_index])]['yaw_rate'])
                        theta, dx, dy = get_combined_transformation_parameter(img.shape[0] / (2 ** i), gridmap_size, frame_rate, vels, yaw_rates)
                    
                    transform_matrix[i, :] = get_STM_matrix(
                        img.shape[0] / (2 ** i), theta, dx, dy)
                if counter % 100 == 0:
                    if counter > 100:
                        CURSOR_UP_ONE = '\x1b[1A'
                        ERASE_LINE = '\x1b[2K'
                        print ""+CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE
                    print str(counter).zfill(5)+"|"+str(len(frames)).zfill(5)+": Saved " + save_path
                np.savez_compressed(save_path, transform_matrix)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--inpath", type=str, dest="frame_folder",
                        required=True, help="Path to directory of the images")
    parser.add_argument("--outpath", type=str, dest="out_path",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--odfile", type=str, dest="odometry_file",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--gridfile", type=str, dest="gridmap_file",
                        required=True, help="Path to directory where images should be saved")
    parser.add_argument("--fr", type=int, dest="frame_rate",
                        default=10, help="Path to directory where images should be saved")
    parser.add_argument("--combFrames", type=int, dest="combined_frames",
                        default=1, help="Number of frames translation which should be combined (for decreasing frame rate)")
    parser.add_argument("--overwrite", type=str2bool, dest="overwrite",
                        default=False, help="Determines whether existing files in --outpath should be overwritten or skipped")

    args = parser.parse_args()
    main(**vars(args))
