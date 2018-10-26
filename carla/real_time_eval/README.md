# CARLA documentation
A short documentation of the CARLA real-time evaluation script.

## Running scripts
The script uses `tf_carla_eval.py` and `preprocessing_situ_all_data.py` to preprocess data. It creates a window showing the next 10 predicted frames in real-time while running Carla. We use the `tkinter` package to produce the output window containing the predictions.

Note: Before running this change line 22 in `tf_carla_eval.py` to reflect the current location.

Copy this directory to:
```
/carla/PythonClient/real_time_eval
```

and run arguments while CARLA is running in server mode with example arguments:
```
run carla_eval_realtime.py --save-run=True --autopilot --same-start=3 --chckpt-loc="<repository location>/large data format/eval_large_data_format/model/"
```

Note: This script uses win32api to read the WASD keys. The code will only run on Linux with the autopilot parameter set.

Important commandline arguments are:
  - synchronous-mode: If the server should run in synchronous or asynchronous mode. Can exhibit unexpected behaviour in asynchronous mode since time stamps are not synced.
  - autopilot: Allows the vehicle to drive using CARLA's autopilot feature for the first couple of frames before the predicted speed and yaw rate are used to control the vehicle. If this is set to false, the user can control the vehicle with the WASD keys.
  - episode: Number of episodes to run.
  - frames: Number of frames per episode.
  - only-predict-drive: Evaluation call internally only returns speed and yaw rate prediction. Does not display grid map and image predictions. Can be used to only test autonomous driving capability.
  - save-run: The episode will be saved and converted to video. The script will create a folder `carla/PythonClient/video_output` and use the `tmpFrameFolder` to store frames before converting them to a mp4 video file. It will save the episode as a video file with the grid map and image predictions and 3rd person view of the vehicle. It will remove the frames of that episode and only keep the video file. Additionally, it will move to the next episode if the vehicle is stuck for 5 consecutive frames. We are using FFmpeg to convert frames to video.
  - dest-path: If save-run is used, this is the location where temporary frames and final videos are stored.
  - same-start: Starts the simulation without NPCs, with sunny weather and at the same location. Can be set to any starting position defined in `/carla/town_positions.pdf`. If set to the default value `-1` the simulation starts at a random location with random weather conditions and 150 vehicles and 100 pedestrians.
  - chckpt-loc: Location where model checkpoint is stored.
  - model-name: Name of the model to load.
