"""
@author davidao

Script to extract DDS information from *$timestamp*vehicle.txt states with following format:
    x [m], y [m], yaw [deg] (grid map center in IDS), speed [m/s], yawrate [deg/s]

Usage:
    python --folder $datapath --save $filename 

Return:
    Gathered list of all DDS states in odometry.txt file
        timestamp [mu s], x [m], y [m], yaw [deg], yawrate [deg/s], speed [m/s]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import argparse
import collections
import os

parser = argparse.ArgumentParser(description='processing odometry files')
parser.add_argument('--folder', default='./',
                    help='data folder path')
parser.add_argument('--save', default='./odometry_t_mus-x_m-y_m-yaw_deg-yr_degs-v_ms.txt',
                    help='save path')

# Extract the parsed input
args = parser.parse_args()
if not os.path.isfile(args.save):
    print("##### Settings #####")
    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args,arg)))
    print("\n")

    print("##### Processing #####")
    all_files_names = glob.glob(args.folder + '/*_ego_vehicle.txt')
    print("=> Detected {} files".format(len(all_files_names)))

    result = {}


    with open(args.save,'w') as w:
        
        print("Storing in this format")
        print("timestamp [mu s], x [m], y [m], yaw [deg], yawrate [deg/s], (grid map center in IDS), speed [m/s]")
        for i, file_name in enumerate(all_files_names):
            with open(file_name,'r') as f:
                timestamp = file_name.rsplit('_',-1)[-3]
                # Logging progress     
                if i % 100 == 0:                                     
                    print("Processing {}".format(file_name))
                    print("Timestamp {}".format(timestamp))
                    print("{} / {}".format(i, len(all_files_names))) 
                lines = [line for line in f]
                state_line = lines[1] # extract second line
                x, y, yaw, v, yaw_rate = state_line.strip().split(' ')
                #print(x, y, yaw, v, yaw_rate)
                result[int(timestamp)] = "{} {} {} {} {} {}\n".format(timestamp, x, y, yaw, yaw_rate, v)
        
        # Sorting the timestamp values
        od = collections.OrderedDict(sorted(result.items()))
        for k,v in od.items():
            w.write(v)
else:
    print("Is already a file: {}\n".format(args.save))