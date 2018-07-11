from __future__ import print_function

import argparse
import logging
import random
import time
import math
import os
import numpy as np
from PIL import Image, ImageDraw

from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

def cart2pol(x, y):
    #rho = np.sqrt(x**2 + y**2) #only need angle
    phi = np.arctan2(y, x)
    #return(rho, phi)
    return(phi)

def agent2world(vehicle, angle):
    loc_x = vehicle.transform.location.x
    loc_y = vehicle.transform.location.y
    ext_x = vehicle.bounding_box.extent.x / 2.0
    ext_y = vehicle.bounding_box.extent.y / 2.0
    bbox = np.array([[ext_x, ext_y],[ext_x, -ext_y],[-ext_x, -ext_y],[-ext_x, ext_y]]).T
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])
    bbox_rot = rotMatrix.dot(bbox)
    return bbox_rot + np.repeat(np.array([[loc_x], [loc_y]]), 4, axis=1)
    
def world2player(polygon, angle, player_transform):
    rotMatrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle),  np.cos(angle)]])
    polygon -= np.repeat(np.array([[player_transform.location.x], [player_transform.location.y]]), 4, axis=1)
    polygon = rotMatrix.dot(polygon)
    return polygon
    
def player2image(polygon, shift, multiplier):
    polygon *= multiplier
    polygon += shift
    return polygon
    
def run_carla_client(args):
    # Here we will run 3 episodes with 300 frames each.
    number_of_episodes = 1
    frames_per_episode = 400

    # We assume the CARLA server is already waiting for a client to connect at
    # host:port. To create a connection we can use the `make_carla_client`
    # context manager, it creates a CARLA client object and starts the
    # connection. It will throw an exception if something goes wrong. The
    # context manager makes sure the connection is always cleaned up on exit.
    with make_carla_client(args.host, args.port) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):

            if args.settings_filepath is None:

                # Create a CarlaSettings object. This object is a wrapper around
                # the CarlaSettings.ini file. Here we set the configuration we
                # want for the new episode.
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=args.synchronous_mode,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=120,
                    NumberOfPedestrians=0,
                    WeatherId=random.choice([1, 3, 7, 8, 14]),
                    QualityLevel=args.quality_level)
                settings.randomize_seeds()

                # Now we want to add a couple of cameras to the player vehicle.
                # We will collect the images produced by these cameras every
                # frame.

                # The default camera captures RGB images of the scene.
                camera0 = Camera('CameraRGB')
                # Set image resolution in pixels.
                camera0.set_image_size(1920, 640)
                # Set its position relative to the car in meters.
                camera0.set_position(2.00, 0, 1.30)
                settings.add_sensor(camera0)

                # Let's add another camera producing ground-truth depth.
                camera1 = Camera('CameraDepth', PostProcessing='Depth')
                camera1.set_image_size(1920, 640)
                camera1.set_position(2.00, 0, 1.30)
                settings.add_sensor(camera1)
                
                #slows down the simulation too much by processing segmentation before saving
                #camera2 = Camera('CameraSegmentation', PostProcessing='SemanticSegmentation')
                #camera2.set_image_size(1920, 640)
                #camera2.set_position(2.00, 0, 1.30)
                #settings.add_sensor(camera2)

                if args.lidar:
                    lidar = Lidar('Lidar32')
                    lidar.set_position(0, 0, 2.50)
                    lidar.set_rotation(0, 0, 0)
                    lidar.set(
                        Channels=32,
                        Range=50,
                        PointsPerSecond=100000,
                        RotationFrequency=10,
                        UpperFovLimit=10,
                        LowerFovLimit=-30)
                    settings.add_sensor(lidar)

            else:

                # Alternatively, we can load these settings from a file.
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            # Now we load these settings into the server. The server replies
            # with a scene description containing the available start spots for
            # the player. Here we can provide a CarlaSettings object or a
            # CarlaSettings.ini file as string.
            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            player_start = random.randint(0, max(0, number_of_player_starts - 1))

            # Notify the server that we want to start the episode at the
            # player_start index. This function blocks until the server is ready
            # to start the episode.
            print('Starting new episode at %r...' % scene.map_name)
            client.start_episode(player_start)
            
            # Start a new episode.
            file_loc = args.file_loc_format.format(episode)
            if not os.path.exists(file_loc):
                os.makedirs(file_loc)
                
            file_loc_tracklets = file_loc + "/tracklets/"
            if not os.path.exists(file_loc_tracklets):
                os.makedirs(file_loc_tracklets)
                
            file_loc_grid = file_loc + "/gridmap/"
            if not os.path.exists(file_loc_grid):
                os.makedirs(file_loc_grid)
                
            print('Data saved in %r' % file_loc)

            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                # Read the data produced by the server this frame.
                measurements, sensor_data = client.read_data()

                # Print some of the measurements.
                print_measurements(measurements)

                # Save the images to disk if requested. Skip first couple of images due to setup time, frame needs to be > 0.
                if args.save_images_to_disk and frame > 9:
                    player_measurements = measurements.player_measurements

                    yaw = player_measurements.transform.rotation.yaw + 180 - yaw_shift
                    if yaw < 0:
                        yaw = yaw + 360;

                    #calculate yaw_rate
                    game_time = np.int64(measurements.game_timestamp)
                    time_diff = (game_time - prev_time) / 1000 # ms -> sec
                    prev_time = game_time
                    if time_diff == 0:
                        yaw_rate = 0
                    else:
                        yaw_rate = (180 - abs(abs(yaw - yaw_old) - 180))/time_diff
                    #print('time_diff: %f, yaw: %f, yaw_old: %f, yaw_rate: %f' % (time_diff, yaw, yaw_old, yaw_rate))
                    yaw_old = yaw

                    with open(file_loc+"odometry_t_mus-x_m-y_m-yaw_deg-yr_degs-v_ms.txt", "a") as text_file:
                        text_file.write("%d %f %f %f %f %f\n" % \
                           (args.start_time+measurements.game_timestamp,
                            player_measurements.transform.location.x - shift_x,
                            player_measurements.transform.location.y - shift_y,
                            yaw,
                            yaw_rate,
                            player_measurements.forward_speed))
                    for name, measurement in sensor_data.items():
                        filename = args.out_filename_format.format(episode, name, frame)
                        if name == 'CameraSegmentation':
                            measurement.save_to_disk_converted(filename)
                        else:
                            measurement.save_to_disk(filename)
                            
                    with open(file_loc_tracklets+str(round(args.start_time+measurements.game_timestamp))+".txt", "w") as text_file:
                        im = Image.new('L', (256*6, 256*6), (127))
                        shift = 256*6/2
                        draw = ImageDraw.Draw(im)
                        for agent in measurements.non_player_agents:
                            if agent.HasField('vehicle'):
                                vehicle = agent.vehicle
                                angle = cart2pol(vehicle.transform.orientation.x, vehicle.transform.orientation.y)
                                text_file.write("%d %f %f %f %f %f\n" % \
                                 (agent.id,
                                  vehicle.transform.location.x,
                                  vehicle.transform.location.y,
                                  angle,
                                  vehicle.bounding_box.extent.x,
                                  vehicle.bounding_box.extent.y))
                                polygon = agent2world(vehicle, angle)
                                polygon = world2player(polygon, math.radians(-yaw), player_measurements.transform)
                                polygon = player2image(polygon, shift, multiplier=25)
                                polygon = [tuple(row) for row in polygon.T]

                                #p1 = (-(loc_x-ext_x)*10+shift,-(loc_y-ext_y)*10+shift)
                                #p2 = (-(loc_x+ext_x)*10+shift,-(loc_y-ext_y)*10+shift)
                                #p3 = (-(loc_x+ext_x)*10+shift,-(loc_y+ext_y)*10+shift)
                                #p4 = (-(loc_x-ext_x)*10+shift,-(loc_y+ext_y)*10+shift)
                                draw.polygon(polygon, 0, 0)
                        im = im.resize((256,256), Image.ANTIALIAS) #if nothing visible try without resize
                        im = im.rotate(imrotate)
                        im.save(file_loc_grid + 'gridmap_'+ str(round(args.start_time+measurements.game_timestamp)) +'_occupancy' + '.png')
                            
                else:
                    # get first values
                    yaw_old = measurements.player_measurements.transform.rotation.yaw + 180
                    imrotate = round(yaw_old)-90
                    if imrotate < 0:
                        imrotate += 360
                    yaw_shift = yaw_old
                    shift_x = measurements.player_measurements.transform.location.x
                    shift_y = measurements.player_measurements.transform.location.y
                    prev_time = np.int64(measurements.game_timestamp)

                # We can access the encoded data of a given image as numpy
                # array using its "data" property. For instance, to get the
                # depth value (normalized) at pixel X, Y
                #
                #     depth_array = sensor_data['CameraDepth'].data
                #     value_at_pixel = depth_array[Y, X]
                #

                # Now we have to send the instructions to control the vehicle.
                # If we are in synchronous mode the server will pause the
                # simulation until we send this control.

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
        '-l', '--lidar',
        action='store_true',
        help='enable Lidar')
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

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.start_time = time.time()*1000

    args.file_loc_format = 'Z:/thesis/carla/run_' + str(int(round(args.start_time))) + '/episode_{:0>4d}/'

    args.out_filename_format = args.file_loc_format + '{:s}/{:0>6d}'

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
