# MBRDNA documentation
A short documentation of the MBRDNA scripts.

Note: These scripts are only used to preprocess the MBRDNA dataset. The resulting dataset can be found at `/mnt/ds3lab-scratch/lucala/datasets/MBRDNA`.

## Setting up MBRDNA
The raw MBRDNA files can be found at `/mnt/ds3lab-scratch/daod/mercedes/ego_vehicle_at_center`. Download and decompress the files.

Ensure the following naming scheme per frame `x`:
  - occupancy: `gridmap_x_occupancy.png`.
  - segmentation: `gridmap_x_stereo_cnn.png`.
  - RGB stereo: `gridmap_x_stereo_img.png`.
  - meta data: `gridmap_x_meta_data.json`. Speed and yaw rate are defined under dynamics.

Note: This naming scheme can be changed in `convert_MB.py` if required.

## Running conversion scripts
Arguments available:
  - file-loc: Location where raw data is stored.
  - storage-loc: Location where to store TFRecords.
  - prefix: String prepended to TFRecord name.
  - samples-per-record: Number of sequences per TFRecord.
  - K: Number of frames to observe before prediction.
  - T: Number of frames to predict.
  - image-size: Size of grid map.
  - seq-length: How many frames per Sequence. Has to be at least K+T+1. Useful when testing different (K,T) combinations.
  - step-size: Number of frames to skip between sequences.

Run conversion script with
```
python convert_MB.py --file-loc="./data/" --storage-loc="./tfrecords/" --prefix="data" --samples-per-record=20 --K=9 --T=10 --image-size=96 --seq-length=20 --step-size=5
```
