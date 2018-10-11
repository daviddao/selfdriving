#CREATES ODOMETRY FILE AND OCCUPANCY MAP FROM KITTI DATA

import numpy as np
import pykitti
import parseTrackletXML as xmlParser
import array
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import collections  as mc
from matplotlib import patches
from tqdm import tqdm
from PIL import Image, ImageDraw
import os
import argparse
import math

# basedir is global and set in main
basedir = '-1'

def load_dataset(date, drive, calibrated=False, frame_range=None):
    """
    Loads the dataset with `date` and `drive`.

    Parameters
    ----------
    date        : Dataset creation date.
    drive       : Dataset drive.
    calibrated  : Flag indicating if we need to parse calibration data. Defaults to `False`.
    frame_range : Range of frames. Defaults to `None`.

    Returns
    -------
    Loaded dataset of type `raw`.
    """
    dataset = pykitti.raw(basedir, date, drive)

    # Load the data
    if calibrated:
        dataset.load_calib()  # Calibration data are accessible as named tuples

    np.set_printoptions(precision=4, suppress=True)
    print('\nDrive: ' + str(dataset.drive))
    print('\nFrame range: ' + str(dataset.frames))

    if calibrated:
        print('\nIMU-to-Velodyne transformation:\n' + str(dataset.calib.T_velo_imu))
        print('\nGray stereo pair baseline [m]: ' + str(dataset.calib.b_gray))
        print('\nRGB stereo pair baseline [m]: ' + str(dataset.calib.b_rgb))

    return dataset


def load_tracklets_for_frames(n_frames, xml_path):
    """
    Loads dataset labels also referred to as tracklets, saving them individually for each frame.

    Parameters
    ----------
    n_frames    : Number of frames in the dataset.
    xml_path    : Path to the tracklets XML.

    Returns
    -------
    Tuple of dictionaries with integer keys corresponding to absolute frame numbers and arrays as values. First array
    contains coordinates of bounding box vertices for each object in the frame, and the second array contains objects
    types as strings.
    """
    tracklets = xmlParser.parseXML(xml_path)

    frame_tracklets = {}
    frame_tracklets_types = {}
    for i in range(n_frames):
        frame_tracklets[i] = []
        frame_tracklets_types[i] = []

    # loop over tracklets
    for i, tracklet in enumerate(tracklets):
        # this part is inspired by kitti object development kit matlab code: computeBox3D
        h, w, l = tracklet.size
        # in velodyne coordinates around zero point and without orientation yet
        trackletBox = np.array([
            [-l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [0.0, 0.0, 0.0, 0.0, h, h, h, h]
        ])
        # loop over all data in tracklet
        for translation, rotation, state, occlusion, truncation, amtOcclusion, amtBorders, absoluteFrameNumber in tracklet:
            # determine if object is in the image; otherwise continue
            if truncation not in (xmlParser.TRUNC_IN_IMAGE, xmlParser.TRUNC_TRUNCATED):
                continue
            # re-create 3D bounding box in velodyne coordinate system
            yaw = rotation[2]  # other rotations are supposedly 0
            assert np.abs(rotation[:2]).sum() == 0, 'object rotations other than yaw given!'
            rotMat = np.array([
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0]
            ])
            cornerPosInVelo = np.dot(rotMat, trackletBox) + np.tile(translation, (8, 1)).T
            frame_tracklets[absoluteFrameNumber] = frame_tracklets[absoluteFrameNumber] + [cornerPosInVelo]
            frame_tracklets_types[absoluteFrameNumber] = frame_tracklets_types[absoluteFrameNumber] + [
                tracklet.objectType]

    return (frame_tracklets, frame_tracklets_types)


def main(date, drive, file_loc, storage_loc):
    global basedir
    basedir = file_loc

    # load raw data
    dataset = load_dataset(date, drive)
    tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), basedir+'{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date, date, drive))

    # extract velocity, position and timestamps
    dataset_velo = list(dataset.velo)
    dataset_timestamps = list(dataset.timestamps)
    dataset_oxts = dataset.oxts

    if not os.path.exists(storage_loc):
        os.makedirs(storage_loc)

    #GPS
    R = 6371000 #radius earth [m]
    curr_oxts = dataset_oxts[0][0]
    shift_x = R * np.cos(curr_oxts.lat) * np.cos(curr_oxts.lon)
    shift_y = R * np.cos(curr_oxts.lat) * np.sin(curr_oxts.lon)
    yaw_shift = 180 * curr_oxts.yaw / math.pi + 180;

    for curr_frame in tqdm(range(len(dataset_velo)-1)):

        # creating grid map
        im = Image.new('L', (256*6, 256*6), (127))
        shift = 256*6/2
        draw = ImageDraw.Draw(im)
        lines = []
        # draw lidar projections
        for i in range(dataset_velo[curr_frame].shape[0]):
            x1, x2 = dataset_velo[curr_frame][i][0]*10+shift, 0+shift
            y1, y2 = -dataset_velo[curr_frame][i][1]*10+shift, -0+shift
            draw.line([(x1,y1),(x2,y2)], fill='white', width=1)

        # draw bounding box polygons
        for t_rects, t_type in zip(tracklet_rects[curr_frame], tracklet_types[curr_frame]):
            vertices = t_rects[[0, 1, 2], :]
            p1 = (vertices[0,1]*10+shift,-vertices[1,1]*10+shift) #left bottom point
            p2 = (vertices[0,2]*10+shift,-vertices[1,2]*10+shift)
            p3 = (vertices[0,3]*10+shift,-vertices[1,3]*10+shift)
            p4 = (vertices[0,0]*10+shift,-vertices[1,0]*10+shift)
            draw.polygon([p1,p2,p3,p4], 0, 0)

        im = im.resize((256,256), Image.ANTIALIAS)

        # extract delta t
        ms = round(dataset_timestamps[curr_frame].timestamp()*1000)
        curr_time = dataset_timestamps[curr_frame].timestamp()
        next_time = dataset_timestamps[curr_frame+1].timestamp()
        time_diff = next_time-curr_time

        # save grid map and blank horizon map (road map)
        im = im.rotate(90)
        im.save(storage_loc + 'gridmap_'+ str(ms) +'_occupancy' + '.png')
        imblack = Image.new('RGB', (256, 256), (255,255,255))
        imblack.save(storage_loc + 'gridmap_'+ str(ms) +'_horizon_map' + '.png')

        # compute yaw, yaw rate, speed and current position
        curr_oxts = dataset_oxts[curr_frame][0]
        next_oxts = dataset_oxts[curr_frame+1][0]
        yaw_next = next_oxts.yaw
        yaw = curr_oxts.yaw
        yaw = 180 * yaw / math.pi + 180 # range [0,360]
        yaw_next = 180 * yaw_next / math.pi + 180 # range [0,360]
        if time_diff == 0:
            yaw_rate = 0
        else:
            yaw_rate = (180 - abs(abs(yaw_next - yaw) - 180))/time_diff

        ids_x = R * np.cos(curr_oxts.lat) * np.cos(curr_oxts.lon)
        ids_y = R * np.cos(curr_oxts.lat) * np.sin(curr_oxts.lon)

        ids_x = ids_x - shift_x
        ids_y = ids_y - shift_y

        speed = np.linalg.norm([curr_oxts.vf, curr_oxts.vl])*np.sign(curr_oxts.vf)
        
        # save odometry in .txt file
        with open(storage_loc+"odometry_t_mus-x_m-y_m-yaw_deg-yr_degs-v_ms.txt", "a") as text_file:
                            text_file.write("%d %f %f %f %f %f\n" % \
                               (ms,ids_x,ids_y,yaw-yaw_shift,yaw_rate,speed))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, dest="date",
                        default='2011_09_26', help="date of data folder (ex. 2011_09_26)")
    parser.add_argument("--drive", type=str, dest="drive",
                        default='0005', help="drive string of data folder (ex. 0005)")
    parser.add_argument("--storage-loc", type=str, dest="storage_loc",
                        default='./processed/', help="location where data will be saved")
    parser.add_argument("--file-loc", type=str, dest="file_loc", default="./data/",
                        help="where is the data located?")
    args = parser.parse_args()
    main(**vars(args))
