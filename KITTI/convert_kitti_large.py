#CONVERTS KITTI TO TFRECORD WITH GRID MAP AND RGB. SEGMENTATION AND DEPTH ARE SET TO BLACK.

from PIL import Image, ImageDraw
import math
import numpy as np
import glob
from tqdm import tqdm
import time
import json
import os
from argparse import ArgumentParser
import preprocessing_situ_all_data
import pykitti
import parseTrackletXML as xmlParser

# Set when calling main
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

def main(drive, date, storage_loc, file_loc, prefix, _samples_per_record, _K, _T, _image_size, _seq_length, _step_size):
    global basedir
    basedir = file_loc

    # setup preprocessing values
    preprocessing_situ_all_data.set_dest_path(storage_loc, _samples_per_record, _K, _T, _image_size, _seq_length, _step_size)
    preprocessing_situ_all_data.update_episode_reset_globals(prefix)
    ind = 0
    dataset = load_dataset(date, drive)
    tracklet_rects, tracklet_types = load_tracklets_for_frames(len(list(dataset.velo)), basedir+'{}/{}_drive_{}_sync/tracklet_labels.xml'.format(date, date, drive))

    # extract velocity, timestamps, coordinates and RGB image
    dataset_velo = list(dataset.velo)
    dataset_timestamps = list(dataset.timestamps)
    dataset_oxts = dataset.oxts
    dataset_rgb = dataset.rgb

    # compute position and initial yaw offset
    R = 6371000 #radius earth [m]
    curr_oxts = dataset_oxts[0][0]
    shift_x = R * np.cos(curr_oxts.lat) * np.cos(curr_oxts.lon)
    shift_y = R * np.cos(curr_oxts.lat) * np.sin(curr_oxts.lon)
    yaw_shift = 180 * curr_oxts.yaw / math.pi + 180;

    for curr_frame in tqdm(range(len(dataset_velo)-1)):

        # create grid map image
        im = Image.new('L', (256*6, 256*6), (127))
        shift = 256*6/2
        draw = ImageDraw.Draw(im)
        lines = []
        # draw lidar projection lines
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

        # resize image to final grid map size
        im = im.resize((256,256), Image.ANTIALIAS)

        # compute delta t
        ms = round(dataset_timestamps[curr_frame].timestamp()*1000)
        curr_time = dataset_timestamps[curr_frame].timestamp()
        next_time = dataset_timestamps[curr_frame+1].timestamp()
        time_diff = next_time-curr_time

        # orientate grid map correctly and create black img for seg. & depth
        gridmap = im.rotate(90)
        imblack = Image.new('RGB', (1920,640), (255,255,255))

        # compute current yaw, yaw rate and speed
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

        speed = np.linalg.norm([curr_oxts.vf, curr_oxts.vl])*np.sign(curr_oxts.vf) #need to flip sign because norm does not account for backward driving case

        # obtain RGB image
        cam2_image, cam3_image = dataset.get_rgb(curr_frame)
        
        # pass information to preprocessing script. will bundle frames to TFRecord once enough are received.
        preprocessing_situ_all_data.main(gridmap,
            np.asarray(cam2_image.resize((1920,640),Image.ANTIALIAS)), np.asarray(imblack),
            np.asarray(imblack),
            yaw_rate, speed)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--date", type=str, dest="date", default="2011_09_11",
                        help="value found on KITTI.")
    parser.add_argument("--drive", type=str, dest="drive", default="0005",
                        help="value found on KITTI.")
    parser.add_argument("--file-loc", type=str, dest="file_loc", default="../data/",
                        help="where is the data located?")
    parser.add_argument("--storage-loc", type=str, dest="storage_loc", default="./tfrecords/",
                        help="where should tfRecords be stored?")
    parser.add_argument("--prefix", type=str, dest="prefix", default="data",
                        help="string prepended to tfRecord name.")
    parser.add_argument("--samples-per-record", type=int, dest="_samples_per_record", default=20,
                        help="Number of sequences per TFRecord.")
    parser.add_argument("--K", type=int, dest="_K", default=9,
                        help="Number of frames to observe before prediction.")
    parser.add_argument("--T", type=int, dest="_T", default=10,
                        help="Number of frames to predict.")
    parser.add_argument("--image-size", type=int, dest="_image_size", default=96,
                        help="Size of grid map.")
    parser.add_argument("--seq-length", type=int, dest="_seq_length", default=40,
                        help="How many frames per Sequence. Has to be at least K+T+1.")
    parser.add_argument("--step-size", type=int, dest="_step_size", default=5,
                        help="Number of frames to skip between sequences.")

    args = parser.parse_args()
    main(**vars(args))
