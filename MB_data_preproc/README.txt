running preprocessing code:

1. ensure all MB data is in one folder where the naming scheme is as follows
  occupancy -> gridmap_x_occupancy.png
  segmentation -> gridmap_x_stereo_cnn.png
  RGB stereo -> gridmap_x_stereo_img.png
  meta data -> gridmap_x_meta_data.json,
      where speed & yaw_rate are defined under dynamics->speed/yawrate->value

  and x is the frame number
  (if this naming scheme is not correct it can be changed in convert_MB.py)

2. run python convert_MB.py
  args:
     --file-loc=<data folder location>
     --storage-loc=<location for folder containing tfRecords>
     --prefix=<string prepended to tfRecord name> (optional)

important note: the first section of the tfRecord name string can be changed,
  but DO NOT change the rest as the string is parsed for information.

additional info. the conversion currently goes through the data as follows:
  - after reading 20 frames bundles them into one sequence
  - discards the first 5 frames and reuses the other 15 frames together with
    5 new frames for the next sequence (5 frame step_size)
  - after collecting 20 sequences (each 20 frames) bundle them into a tfRecord
  - a tfRecord now contains 20 sequences with each 20 frames:
    1 frame is made up of:
      input_seq: gridmap input frames
      target_seq: gridmap ground truth frames
      maps: horizon map, not used (white img, compatibility with carla)
      tf_matrix: transformation matrix encoding speed and yawrate
      rgb: RGB image of front view
      segmentation: segmentation image of front view
      depth: depth image of front view (black img, compatibility with MB data)
      direction: 2 bit value of current direction based on yaw_rate
        (straight, left, right)
