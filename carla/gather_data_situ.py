from __future__ import print_function

import argparse
import logging
import random
import time
import math
import os
import numpy as np
from PIL import Image, ImageDraw
import preprocessing_situ
import preprocessing_situ_all_data

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

def cart2pol(x, y):
    phi = np.arctan2(y, x)
    return phi

# transform agent coordinate to world coordinate
def agent2world(vehicle, angle):
    loc_x = vehicle.transform.location.x
    loc_y = vehicle.transform.location.y
    ext_x = vehicle.bounding_box.extent.x / 2.0
    ext_y = vehicle.bounding_box.extent.y / 2.0
    bbox = np.array([[ext_x, ext_y],[ext_x, -ext_y],[-ext_x, -ext_y],[-ext_x, ext_y]]).T
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])
    bbox_rot = rotMatrix.dot(bbox)
    return bbox_rot + np.repeat(np.array([[loc_x], [loc_y]]), 4, axis=1)

# transform world coordinate to player coordinate
def world2player(polygon, angle, player_transform):
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])
    polygon -= np.repeat(np.array([[player_transform.location.x], [player_transform.location.y]]), 4, axis=1)
    polygon = rotMatrix.dot(polygon)
    return polygon

# transform player coordinates to image coordinates
def player2image(polygon, shift, multiplier):
    polygon *= multiplier
    polygon += shift
    return polygon

# process player odometry
def process_odometry(measurements, yaw_shift, yaw_old, prev_time):
    player_measurements = measurements.player_measurements
    yaw = ((player_measurements.transform.rotation.yaw - yaw_shift - 180) % 360) - 180

    #calculate yaw_rate
    game_time = np.int64(measurements.game_timestamp)
    time_diff = (game_time - prev_time) / 1000 # ms -> sec
    prev_time = game_time
    if time_diff == 0:
        yaw_rate = 0
    else:
        yaw_rate = (180 - abs(abs(yaw - yaw_old) - 180))/time_diff * np.sign(yaw-yaw_old)

    return yaw, yaw_rate, player_measurements.forward_speed, prev_time

def run_carla_client(args):
    # Here we will run args._episode episodes with args._frames frames each.
    number_of_episodes = args._episode
    frames_per_episode = args._frames

    # create the carla client
    with make_carla_client(args.host, args.port, timeout=100000) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):

            if args.settings_filepath is None:

                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=args.synchronous_mode,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=150,
                    NumberOfPedestrians=100,
                    WeatherId=random.choice([1, 3, 7, 8, 14]),
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

                camera0 = Camera('CameraRGB')
                camera0.set_image_size(1920, 640)
                camera0.set_position(2.00, 0, 1.30)
                settings.add_sensor(camera0)

                camera1 = Camera('CameraDepth', PostProcessing='Depth')
                camera1.set_image_size(1920, 640)
                camera1.set_position(2.00, 0, 1.30)
                settings.add_sensor(camera1)

                camera2 = Camera('CameraSegmentation', PostProcessing='SemanticSegmentation')
                camera2.set_image_size(1920, 640)
                camera2.set_position(2.00, 0, 1.30)
                settings.add_sensor(camera2)

            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))

            # create folder for current episode
            file_loc = args.file_loc
            if not os.path.exists(file_loc):
                os.makedirs(file_loc)

            #set destination path for the tfrecords
            preprocessing_situ_all_data.set_dest_path(file_loc, args._samples_per_record, args._K, args._T, args._image_size, args._seq_length, args._step_size)

            print('Starting new episode at %r...' % scene.map_name)
            client.start_episode(player_start)

            # take unique time snapshot used for naming
            prefix = str(int(round(args.start_time))) + '_' + str(episode)

            #update the prefix of the tfrecord name and reset the globals when starting a new episode
            preprocessing_situ_all_data.update_episode_reset_globals(prefix)

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                measurements, sensor_data = client.read_data()

                #discard episode if misbehaviour flag is set and a form of collision is detected
                if args.no_misbehaviour:
                    player_measurements = measurements.player_measurements
                    col_cars=player_measurements.collision_vehicles
                    col_ped=player_measurements.collision_pedestrians
                    col_other=player_measurements.collision_other
                    if col_cars + col_ped + col_other > 0:
                        print("MISBEHAVIOUR DETECTED! Discarding Episode... \n")
                        break

                print_measurements(measurements)

                # Save the images to disk if requested. Skip first couple of images due to setup time.
                if args.save_images_to_disk and frame > 19:
                    player_measurements = measurements.player_measurements

                    for name, measurement in sensor_data.items():
                        if name == 'CameraDepth':
                            depth = measurement.return_depth_map()
                        if name == 'CameraSegmentation':
                            segmentation = measurement.return_segmentation_map()
                        if name == 'CameraRGB':
                            rgb = measurement.return_rgb()

                    yaw, yaw_rate, speed, prev_time = process_odometry(measurements, yaw_shift, yaw_old, prev_time)
                    yaw_old = yaw

                    #generate image with bounding boxes, will be transformed to gridmaps in preprocessing
                    im = Image.new('L', (256*6, 256*6), (127))
                    shift = 256*6/2
                    draw = ImageDraw.Draw(im)
                    for agent in measurements.non_player_agents:
                        if agent.HasField('vehicle') or agent.HasField('pedestrian'):
                            participant = agent.vehicle if agent.HasField('vehicle') else agent.pedestrian
                            angle = cart2pol(participant.transform.orientation.x, participant.transform.orientation.y)
                            polygon = agent2world(participant, angle)
                            polygon = world2player(polygon, math.radians(-yaw), player_measurements.transform)
                            polygon = player2image(polygon, shift, multiplier=25)
                            polygon = [tuple(row) for row in polygon.T]
                            draw.polygon(polygon, 0, 0)
                    im = im.resize((256,256), Image.ANTIALIAS)
                    im = im.rotate(imrotate)

                    #pass frame by frame to the preprocessor, it buffers these and stores them as tfrecords
                    if not args.all_data:
                        preprocessing_situ.main(im, -yaw_rate, speed)
                    else:
                        preprocessing_situ_all_data.main(im, rgb, depth, segmentation, -yaw_rate, speed)

                else:
                    # get values while frame < 20 to be used when frame == 20
                    yaw_shift = measurements.player_measurements.transform.rotation.yaw
                    yaw_old = ((yaw_shift - 180) % 360) - 180
                    imrotate = round(yaw_old)+90
                    shift_x = measurements.player_measurements.transform.location.x
                    shift_y = measurements.player_measurements.transform.location.y
                    prev_time = np.int64(measurements.game_timestamp)

                if not args.autopilot:

                    client.send_control(
                        steer=random.uniform(-1.0, 1.0),
                        throttle=0.5,
                        brake=0.0,
                        hand_brake=False,
                        reverse=False)

                else:

                    # Together with the measurements, the server has sent the
                    # control that the in-game autopilot would do this frame. We
                    # can enable autopilot by sending back this control to the
                    # server. We can modify it if wanted, here for instance we
                    # will add some noise to the steer.

                    control = measurements.player_measurements.autopilot_control
                    control.steer += random.uniform(-0.01, 0.01)
                    client.send_control(control)


def print_measurements(measurements):
    number_of_agents = len(measurements.non_player_agents)
    player_measurements = measurements.player_measurements
    message = 'Vehicle at ({pos_x:.1f}, {pos_y:.1f}), '
    message += '{speed:.0f} km/h, '
    message += 'Collision: {{vehicles={col_cars:.0f}, pedestrians={col_ped:.0f}, other={col_other:.0f}}}, '
    message += '{other_lane:.0f}% other lane, {offroad:.0f}% off-road, '
    message += '({agents_num:d} non-player agents in the scene)'
    message = message.format(
        pos_x=player_measurements.transform.location.x,
        pos_y=player_measurements.transform.location.y,
        speed=player_measurements.forward_speed * 3.6, # m/s -> km/h
        col_cars=player_measurements.collision_vehicles,
        col_ped=player_measurements.collision_pedestrians,
        col_other=player_measurements.collision_other,
        other_lane=100 * player_measurements.intersection_otherlane,
        offroad=100 * player_measurements.intersection_offroad,
        agents_num=number_of_agents)
    print_over_same_line(message)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-i', '--images-to-disk',
        action='store_true',
        dest='save_images_to_disk',
        help='save images (and Lidar data if active) to disk')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '-s', '--synchronous-mode',
        dest='synchronous_mode',
        default=True,
        help='Synchronous or Asynchronous mode?')
    argparser.add_argument(
        '-d', '--data',
        dest='all_data',
        default=False,
        help='Write only gridmap or also RGB/Segmentation/Depth into TFRecord?')
    argparser.add_argument(
        '-dp', '--dest-path',
        dest='dest_path',
        default='Z:/thesis/carla/',
        help='Where to store generated data?')
    argparser.add_argument(
        '-n', '--no-misbehaviour',
        dest='no_misbehaviour',
        default=True,
        help='Should Episode be discarded if violation was detected?')
    argparser.add_argument("--episode", type=int, dest="_episode", default=100,
                        help="Number of episodes to run.")
    argparser.add_argument("--frames", type=int, dest="_frames", default=500,
                        help="Number of frames per episode.")
    argparser.add_argument("--samples-per-record", type=int, dest="_samples_per_record", default=20,
                        help="Number of sequences per TFRecord.")
    argparser.add_argument("--K", type=int, dest="_K", default=9,
                        help="Number of frames to observe before prediction.")
    argparser.add_argument("--T", type=int, dest="_T", default=10,
                        help="Number of frames to predict.")
    argparser.add_argument("--image-size", type=int, dest="_image_size", default=96,
                        help="Size of grid map.")
    argparser.add_argument("--seq-length", type=int, dest="_seq_length", default=40,
                        help="How many frames per Sequence. Has to be at least K+T+1.")
    argparser.add_argument("--step-size", type=int, dest="_step_size", default=5,
                        help="Number of frames to skip between sequences.")

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.start_time = time.time()*1000

    args.file_loc = args.dest_path + 'tf_records/'

    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(1)


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
