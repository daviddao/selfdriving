from __future__ import print_function

import argparse
import logging
import random
import time
import math
import os
import glob
import numpy as np
from collections import namedtuple
from PIL import Image, ImageDraw
import preprocessing_situ
import preprocessing_situ_all_data

from carla.client import make_carla_client, VehicleControl
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

import tf_carla_eval

# needed for key press registration in windows when running manual control
try:
    import win32api
except:
    print("Couldn't import win32api. Can only run with autopilot set.")

rangeThreshold = 50
rangeThresholdSq = rangeThreshold**2

# speed up grid map creation by discarding all participants outside of grid map range
def participantInRange(partLoc, plLoc):
    return (((partLoc.x - plLoc.x)**2 + (partLoc.y - plLoc.y)**2) <= rangeThresholdSq)

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
    
    # set speed and yaw rate variable to default
    speedyawrate = np.nan

    #call init in eval file to load model
    checkpoint_dir_loc = args.chckpt_loc
    prefix = args.model_name
    tf_carla_eval.init(checkpoint_dir_loc, prefix)

    # create the carla client
    with make_carla_client(args.host, args.port, timeout=100000) as client:
        print('CarlaClient connected')

        for episode in range(0, number_of_episodes):

            if args.settings_filepath is None:

                # if same starting position arg set use same weather
                if args._sameStart > -1:
                    choice = 1
                    nrVehicles = 0
                    nrPedestrians = 0
                else:
                    choice = random.choice([1, 3, 7, 8, 14])
                    nrVehicles = 150
                    nrPedestrians = 100
                    
                settings = CarlaSettings()
                settings.set(
                    SynchronousMode=args.synchronous_mode,
                    SendNonPlayerAgentsInfo=True,
                    NumberOfVehicles=nrVehicles,
                    NumberOfPedestrians=nrPedestrians,
                    WeatherId=choice,
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
                
                if args._save:
                    camera3 = Camera('CameraRGB_top', PostProcessing='SceneFinal')
                    camera3.set_image_size(800, 800)
                    camera3.set_position(-6.0, 0, 4.0)
                    camera3.set_rotation(yaw=0, roll=0, pitch=-10)
                    settings.add_sensor(camera3)
                    
            else:
                with open(args.settings_filepath, 'r') as fp:
                    settings = fp.read()

            scene = client.load_settings(settings)

            # Choose one player start at random.
            number_of_player_starts = len(scene.player_start_spots)
            if args._sameStart > -1:
                player_start = args._sameStart
            else:
                player_start = random.randint(0, max(0, number_of_player_starts - 1))

            print('Starting new episode at %r...' % scene.map_name)
            client.start_episode(player_start)
            
            if args._save:
                # initialize stuck variable. if car does not move after colliding for x frames we restart episode.
                nrStuck = 0
                # maximum number of times we can get stuck before restarting
                maxStuck = 5
                
                # last location variable to keep track if car is changing position
                last_loc = namedtuple('last_loc', 'x y z')
                last_loc.__new__.__defaults__ = (0.0, 0.0, 0.0)
                last_loc.x = -1
                last_loc.y = -1
                
                # delete frames of previous run to not interfere with video creation
                for rmf in glob.glob(args.file_loc+'tmpFrameFolder/frame*'):
                    os.remove(rmf)
            
            # Iterate every frame in the episode.
            for frame in range(0, frames_per_episode):

                measurements, sensor_data = client.read_data()
                
                # if we are saving video of episode move to next episode if we get stuck
                if args._save:
                    player_measurements = measurements.player_measurements
                    col_other=player_measurements.collision_other
                    col_cars=player_measurements.collision_vehicles
                    curr_loc = player_measurements.transform.location
                    if player_measurements.forward_speed <= 0.05 and sqrdist(last_loc,curr_loc) < 2 and frame > 30:
                        if nrStuck == maxStuck:
                            print("\nWe are stuck! Writing to video and restarting.")
                            #args._sameStart += 1
                            break
                        else:
                            print("Stuck: "+str(nrStuck)+"/"+str(maxStuck))
                            nrStuck += 1
                    last_loc = curr_loc

                # Skip first couple of images due to setup time.
                if frame > 19:
                    player_measurements = measurements.player_measurements

                    for name, curr_measurements in sensor_data.items():
                        if name == 'CameraDepth':
                            depth = curr_measurements.return_depth_map()
                        if name == 'CameraSegmentation':
                            segmentation = curr_measurements.return_segmentation_map()
                        if name == 'CameraRGB':
                            rgb = curr_measurements.return_rgb()

                    yaw, yaw_rate, speed, prev_time = process_odometry(measurements, yaw_shift, yaw_old, prev_time)
                    yaw_old = yaw

                    im = Image.new('L', (256*6, 256*6), (127))
                    shift = 256*6/2
                    draw = ImageDraw.Draw(im)

                    gmTime = time.time()
                    for agent in measurements.non_player_agents:
                        if agent.HasField('vehicle') or agent.HasField('pedestrian'):
                            participant = agent.vehicle if agent.HasField('vehicle') else agent.pedestrian
                            if not participantInRange(participant.transform.location, player_measurements.transform.location):
                                continue
                            angle = cart2pol(participant.transform.orientation.x, participant.transform.orientation.y)
                            polygon = agent2world(participant, angle)
                            polygon = world2player(polygon, math.radians(-yaw), player_measurements.transform)
                            polygon = player2image(polygon, shift, multiplier=25)
                            polygon = [tuple(row) for row in polygon.T]

                            draw.polygon(polygon, 0, 0)

                    im = im.resize((256,256), Image.ANTIALIAS)
                    im = im.rotate(imrotate)
                    gmTime = time.time() - gmTime
                    if not args.all_data:
                        print("only all data supported for now")
                        exit()
                    else:
                        if args._opd:
                            ppTime, evTime, imTime, tkTime, speedyawrate, direction = tf_carla_eval.eval_only_drive(im, rgb, depth, segmentation, -yaw_rate, speed)
                        else:
                            ppTime, evTime, imTime, tkTime, speedyawrate, direction, imReady, im = tf_carla_eval.eval(im, rgb, depth, segmentation, -yaw_rate, speed)
                            if args._save and imReady:
                                # append frame to array for video output
                                for name, curr_measurements in sensor_data.items():
                                    if name == 'CameraRGB_top':
                                        filename = args.file_loc+"tmpFrameFolder/frame"+str(frame).zfill(4)+".jpg"
                                        Image.fromarray(np.concatenate([im,curr_measurements.return_rgb()],1)).save(filename)
                                        break
                                
                        #printTimePerEval(gmTime, ppTime, evTime, imTime, tkTime)

                else:
                    # get first values
                    yaw_shift = measurements.player_measurements.transform.rotation.yaw
                    yaw_old = ((yaw_shift - 180) % 360) - 180
                    imrotate = round(yaw_old)+90
                    shift_x = measurements.player_measurements.transform.location.x
                    shift_y = measurements.player_measurements.transform.location.y
                    prev_time = np.int64(measurements.game_timestamp)

                if not args.autopilot:
                    control = getKeyboardControl()
                else:
                    # values are nan while script is gathering frames before first prediction
                    if np.any(np.isnan(speedyawrate)):
                        control = measurements.player_measurements.autopilot_control
                        control.steer += random.uniform(-0.01, 0.01)
                    else:
                        control = VehicleControl()
                        # speedyawrate contains T entries, first entry is first prediction
                        dirAvgEncoding = np.mean(np.squeeze(np.array(direction)),0)
                        speedyawrate = np.asarray(speedyawrate)
                        steerAvgEncoding = np.mean(np.squeeze(speedyawrate),0)
                        control.throttle = mapThrottle(player_measurements, speedyawrate)
                        control.brake = int(control.throttle == -1)
                        control.steer = mapSteer(steerAvgEncoding[1])
                        #printSteering(measurements.player_measurements.forward_speed,
                        #              measurements.player_measurements.autopilot_control.throttle,
                        #              measurements.player_measurements.autopilot_control.steer,
                        #              speedyawrate, control.throttle, control.steer, 
                        #              dirAvgEncoding, steerAvgEncoding)

                client.send_control(control)
            if args._save:
                print("\nConverting frames to video...")
                os.system("ffmpeg -r 10 -f image2 -start_number 20 -i {}frame%04d.jpg -vcodec libx264 -crf 15 -pix_fmt yuv420p {}".format(args.file_loc+"tmpFrameFolder/", args.file_loc+"video"+str(time.time()*1000)+".mp4"))
                print("Finished conversion.")
                
# break if predicted speed is negative, use throttle 0/1 to obtain predicted speed
def mapThrottle(player_measurements, speedyawrate):
    if speedyawrate[0][0][0] > 0:
        return (player_measurements.forward_speed < speedyawrate[0][0][0]).astype(int)
    else:
        return -1

# remove noisy prediction by not steering if below |1|
def mapSteer(steer):
    if steer <= 0.25 and steer >= -0.25:
        return 0
    else:
        return steer
    
def getKeyboardControl():
    control = VehicleControl()
    if win32api.GetAsyncKeyState(ord('A')):
        control.steer = -1.0
    if win32api.GetAsyncKeyState(ord('D')):
        control.steer = 1.0
    if win32api.GetAsyncKeyState(ord('W')):
        control.throttle = 1.0
    if win32api.GetAsyncKeyState(ord('S')):
        control.brake = 1.0
    return control

def sqrdist(loc1, loc2):
    return (loc1.x-loc2.x)**2 + (loc1.y-loc2.y)**2

def printMeasurements(measurements):
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

def printTimePerEval(time, ppTime, evTime, imTime, tkTime):
    message = 'Time per gridmap gen: {td:.3f}, '
    message += 'Time per preprocess: {pp:.3f}, '
    message += 'Time per eval: {ev:.3f}, '
    message += 'Time per image: {im:.3f}, '
    message += 'Time per tk: {tk:.3f}'
    message = message.format(td=time, pp=ppTime, ev=evTime, im=imTime, tk=tkTime)
    print_over_same_line(message)
    
def printSteering(correctSpeed, correctThrottle, correctSteer, sy, throttle, steer, direction, predictedSteer):
    message = '(correct,prediction) Speed: ({cs: .3f}, {ps: .3f}), '
    message += 'throttle: ({ct: .3f}, {pt: .3f}), '
    message += 'steer: ({cst: .3f}, {pst: .3f}), '
    message += 'predicted mean speed: {p1: .3f}, '
    message += 'predicted mean yaw rate {p2: .3f}, predicted mean binary direction ({pb1: .3f}, {pb2: .3f}), '
    message += ' Steering Prediction:{s1: .3f},{s2: .3f},{s3: .3f},{s4: .3f},{s5: .3f},{s6: .3f}'
    message = message.format(cs=correctSpeed, ps=sy[0][0][0], ct=correctThrottle,
                             pt=throttle, cst=correctSteer, pst=steer,
                             p1=predictedSteer[0], p2=predictedSteer[1],
                             pb1=direction[0], pb2 = direction[1],s1=sy[0][0][1],s2=sy[1][0][1],s3=sy[2][0][1],
                             s4=sy[3][0][1],s5=sy[4][0][1],s6=sy[5][0][1])
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
        default=True,
        help='Show only gridmap or also RGB/Segmentation/Depth? (deprecated)')
    argparser.add_argument(
        '-dp', '--dest-path',
        dest='dest_path',
        default='./',
        help='Where to store generated data?')
    argparser.add_argument("--episode", type=int, dest="_episode", default=100,
                        help="Number of episodes to run.")
    argparser.add_argument("--frames", type=int, dest="_frames", default=1000,
                        help="Number of frames per episode.")
    argparser.add_argument("--only-predict-drive", type=bool, dest="_opd", default=False,
                        help="Only evaluate for driving parameters (throttle, yaw) or also grid map and image channels.")
    argparser.add_argument("--save-run", type=bool, dest="_save", default=False,
                        help="Save run as video sequence?")
    argparser.add_argument("--same-start", type=int, dest="_sameStart", default=-1,
                        help="Start episode at same location everytime.")
    argparser.add_argument("--chckpt-loc", type=str, dest="chckpt_loc", default="../../large data format/eval_large_data_format/model/",
                        help="Location where model checkpoint is stored.")
    argparser.add_argument("--model-name", type=str, dest="model_name", default="64k_noTM_speedyaw_onlyCARLA_image_size=96_K=9_T=10_seqsteps=1_batch_size=4_alpha=1.001_beta=0.0_lr_G=0.0001_lr_D=0.0001_d_in=20_selu=True_comb=False_predV=-1", help="Name of model to load.")

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    args.start_time = time.time()*1000

    args.file_loc = args.dest_path + 'video_output/'
    
    if args._save and not os.path.exists(args.file_loc):
        os.makedirs(args.file_loc)
        os.makedirs(args.file_loc+'tmpFrameFolder/')

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
