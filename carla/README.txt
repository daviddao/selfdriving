After installing Carla:
Using the Unreal Editor build a standalone version of the client. (Explained in Carla Docs)
Copy arguments -windowed -resX=800 -resY=600 -carla-server -fps=10 --carla-settings="path/to/CarlaSettings.ini" into startup

Copy Python files to /carla/PythonClient and execute there while Carla server is running.

Rundown of important files:
CarlaTestPython.ipynb - jupyter notebook used to run python scripts

DQNcarCarla.py - Deep-Q Network RL test for Carla, taken from Airsim Simulator.

gather_data.py - gathers frames of RGB, depth, lidar and occupancy and save to --file-location

gather_data_situ.py - gathers frames of RGB, depth, segmentation and occupancy. Uses preprocessing script to convert data into TFRecord.

preprocessing_situ.py/preprocessing_situ_all_data.py - takes frames and bundles them into TFRecord, _all_data.py also takes RGB/depth/segmentation frames unlike other version that only converts grid maps.

real_time_eval folder - Creates window showing next 10 predicted frames in realtime while running Carla. Run carla_eval_realtime.py, uses tf_carla_eval.py and preprocessing_situ_all_data.py to preprocess data. Note: model location hardcoded in carla_eval_realtime.py line 75.

carla_intersection_locations.txt - values extracted from town_positions that can be used as starting positions for the car. the simulation will start with the vehicle driving towards an intersection.
