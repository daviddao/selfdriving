#generate odometry file

import glob
import os
import json
import collections
from tqdm import tqdm
import argparse

#dir_loc = "F:/selfdriving-data/20180117_1555_cw_big_loop_sunnyvale_ave_traffic_lights_annotation/*.json" #need to change slash in linux
#out_loc = "F:/selfdriving-data/"
#out_name = "20180117_1555_odometry.txt"
#out_file = out_loc+out_name

parser = argparse.ArgumentParser(description='processing odometry files')
parser.add_argument('--folder', default='./',
                    help='data folder path')
parser.add_argument('--save', default='./odometry_t_mus-x_m-y_m-yaw_deg-yr_degs-v_ms.txt',
                    help='save path')

# Extract the parsed input
args = parser.parse_args()

if not os.path.isfile(args.save):
    result = {}

    with open(args.save,'w') as w:
        
        print("Storing in this format")
        print("timestamp [mu s], x [m], y [m], yaw [deg], yawrate [deg/s], (grid map center in IDS), speed [m/s]")
        for f in tqdm(glob.glob(args.folder+"*.json")):
            #print(f)
            with open(f) as j:
                try:
				    data = json.load(j)
                    timestamp = data['timestamp']
                    speed = data['dynamics']['speed']['value']
                    yaw_rate = data['dynamics']['yawrate']['value']
                    x = data['ids']['x']['value']
                    y = data['ids']['y']['value']
                    yaw = data['ids']['yaw']['value']
                    #print(timestamp, x, y, yaw, speed, yaw_rate)
                    result[int(timestamp)] = "{} {} {} {} {} {}\n".format(timestamp, x, y, yaw, yaw_rate, speed)
                except:
                    continue
        
        # Sorting the timestamp values
        od = collections.OrderedDict(sorted(result.items()))
        for k,v in od.items():
            w.write(v)
else:
    print("Is already a file: {}\n".format(args.save))