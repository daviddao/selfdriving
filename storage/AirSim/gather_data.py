import setup_path 
import airsim
import pprint
import math
import time
import os
import numpy as np
import cv2

# connect to the AirSim simulator 
client = airsim.CarClient()
client.confirmConnection()
client.enableApiControl(False) #need control over vehicle
car_controls = airsim.CarControls()

start = time.time()
last_time = start

msRounded = int(start*1000)

_, _, yaw_old = airsim.to_eularian_angles(client.getCarState().kinematics_estimated.orientation)

file_loc = 'C:/Users/lanze/Documents/AirSim/generating_data_python/'

if not os.path.exists(file_loc+ 'run_' + str(msRounded)):
    os.makedirs(file_loc+ 'run_' + str(msRounded))

file_loc += 'run_' + str(msRounded) + '/'

pp = pprint.PrettyPrinter(indent=4)

airsim.wait_key('Press any key to get camera parameters')
for camera_id in range(2):
    camera_info = client.simGetCameraInfo(str(camera_id))
    print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))

#time.sleep(1)
#airsim.wait_key('Press any key to start recording') doesn't work

idx = 0
while True:
    # get state of the car
    car_state = client.getCarState()
    curr_speed = car_state.speed
    curr_gear = car_state.gear
    pos = car_state.kinematics_estimated.position
    orientation = car_state.kinematics_estimated.orientation

    pitch, roll, yaw = airsim.to_eularian_angles(orientation)
    milliseconds = int(round(time.time()* 1000))

    yaw = 180 * yaw / math.pi;
    if yaw < 0:
        yaw = yaw + 360;


    curr_time = time.time() #in seconds
    time_diff = curr_time - last_time
    last_time = curr_time
    yaw_rate = (yaw - yaw_old)/time_diff
    yaw_old = yaw

    with open(file_loc+"odometry_t_mus-x_m-y_m-yaw_deg-yr_degs-v_ms.txt", "a") as text_file:
        text_file.write("%s %f %f %f %f %f\n" % \
       (milliseconds, pos.x_val, pos.y_val, yaw, yaw_rate, curr_speed))

    print("%s %f %f %f %f %f" % \
       (milliseconds, pos.x_val, pos.y_val, yaw, yaw_rate, curr_speed))

    
    time.sleep(1)

    # get camera images from the car
    responses = client.simGetImages([
        airsim.ImageRequest(0, airsim.ImageType.DepthVis, True),  #depth visualization image
        airsim.ImageRequest(0, airsim.ImageType.DisparityNormalized, True), #depth in perspective projection
        airsim.ImageRequest(0, airsim.ImageType.Segmentation, False, False),
        airsim.ImageRequest(0, airsim.ImageType.Scene)]) #scene vision image in png format

    #print('Retrieved images: %d', len(responses))

    for response in responses:
        filename = file_loc + 'data_' + str(milliseconds) + '_'

        if response.pixels_as_float:
            #print("Type %d, size %d" % (response.image_type, len(response.image_data_float)))
            #airsim.write_pfm(os.path.normpath(filename + response.ImageType + '_depth.pfm'), airsim.get_pfm_array(response))

            depth = airsim.get_pfm_array(response)

            depth = np.clip(depth, 0, 1)

            #if disparity need to divide by max value
            if response.image_type == 4:
                depth /= np.max(depth)
            else:
                mean = np.mean(depth)
                shift = 0.5 - mean
                depth += shift

            depth *= 255
            cv2.imwrite(os.path.normpath(filename + str(response.image_type) + '_depth.png'), np.uint8(np.clip(depth, 0, 255)))

        elif response.compress: #png format
            #print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            airsim.write_file(os.path.normpath(filename + str(response.image_type) + '_scene.png'), response.image_data_uint8)
        else: #uncompressed array
            #print("Type %d, size %d" % (response.image_type, len(response.image_data_uint8)))
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8) #get numpy array
            img_rgba = img1d.reshape(response.height, response.width, 4) #reshape array to 4 channel image array H X W X 4
            img_rgba = np.flipud(img_rgba) #original image is flipped vertically
            airsim.write_png(os.path.normpath(filename + str(response.image_type) + '_segmentation.png'), img_rgba) #write to png 

    idx = idx+1


#restore to original state
client.reset()

client.enableApiControl(False)


            
