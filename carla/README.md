# CARLA documentation
A short documentation of the CARLA scripts.

# Setting up CARLA
Follow the instructions for [Windows](https://carla.readthedocs.io/en/latest/how_to_build_on_windows/) or for [Linux](https://carla.readthedocs.io/en/latest/how_to_build_on_linux/), if problems arise their [discord](https://discordapp.com/invite/8kqACuC) channel is a good place to ask for help.

Using the Unreal Editor build a standalone version of the client. When starting the standalone version we used the following arguments:

  - windowed: Fullscreen or windowed mode
  - resX: Resolution on X-axis
  - resY: Resolution on Y-axis
  - carla-server: Scripts do not work without the simulator running in server mode.
  - fps: Simulation speed, for our work we set it to 10.
  - carla-settings: Takes the path to the `CarlaSettings.ini` file.

# CARLA scripts
Copy the python scripts to the subdirectory:
```
/carla/PythonClient
```
Execute the scripts while the CARLA server is running. Make sure that the port set in the `CarlaSettings.ini` and in the scripts match and are not already in use.

## Scripts
  - CarlaRunPython: Jupyter notebook used to run python scripts.
  - gather_data: gathers frames of RGB, depth, lidar and occupancy and saves to `--dest-path`.
  - gather_data_situ: Gathers frames of RGB, depth, segmentation and occupancy. Uses preprocessing script to immediately convert data into TFRecord.
  - preprocessing_situ: Helper scripts for `gather_data_situ.py`. Takes frames and bundles them into TFRecord. This version only converts grid maps.
  - preprocessing_situ_all_data: Helper scripts for `gather_data_situ.py`. Converts all data (RGB/depth/segmentation/grid map).
  - DQNcarCarla.py: Deep-Q Network RL test for Carla, taken from AirSim Simulator.
  - carla_intersection_locations: Text file containing values extracted from `town_positions` with `view_start_positions.py` that can be used as starting positions for the car. The simulation will then start with the vehicle driving towards a random intersection.
  - **real time eval**: Folder containing scripts to run real-time evaluation. Run `carla_eval_realtime.py`, uses `tf_carla_eval.py` and `preprocessing_situ_all_data.py` to preprocess data. Creates window showing next 10 predicted frames in real-time while running Carla. Note: Before running this change line 22 in `tf_carla_eval.py` to reflect the current location.

## Important arguments for data gathering scripts
  - autopilot: We want the vehicle to be autopiloted while gathering data.
  - images-to-disk: We want to save the gathered data to disk.
  - carla-settings: Similar file to what is passed to server. We do not use it because the values are hardcoded in the script. Contains information such as number of pedestrians, vehicles, weather condition, etc.
  - synchronous_mode: We want the script and server to run in synchronous mode so that the simulation stops at every iteration and waits for the script to respond. This allows the time variable to behave physically correct and the script to handle every received frame.
  - data: Store only grid maps or also RGB, segmentation and depth.
  - dest-path: Location to store generated data.
  - no-misbehaviour: Episode will be discarded if violation is detected (collision of any sorts).

Run data gathering code while CARLA server is running with
```
python gather_data_situ.py --autopilot --images-to-disk --data=True --dest-path='./output/'
```
