# KITTI documentation
A short documentation of the KITTI scripts.

Note: These scripts are only used to preprocess the KITTI dataset. The resulting dataset can be found at `/mnt/ds3lab-scratch/lucala/datasets/KITTI`.

## Setting up KITTI
Download the raw KITTI files [here](http://www.cvlibs.net/datasets/kitti/raw_data.php). Download the synced+rectified data and the tracklets. We only download the data that has tracklets, since we need the bounding box information for the grid maps.

Note: For more information on how the KITTI data is stored please see the `kitti_readme.txt`.

## Preprocessing KITTI
To run the KITTI scripts we use the [pykitti](https://github.com/utiasSTARS/pykitti) package to help load the data and `parseTrackletXML.py` to parse the tracklets. The pykitti package needs to be installed before running the scripts.

The two conversion scripts are
  - kitti_converter: Converts KITTI data to odometry.txt and occupancy maps.
  - convert_kitti_large: Converts KITTI data to TFRecord directly. Includes grid map, RGB and black image for segmentation and depth.

Note: Before running the conversion scripts update the `basedir` at the beginning of the file to reflect the location of the raw KITTI data.

Script arguments are
  - date: Part of the KITTI naming scheme.
  - drive: Part of the KITTI naming scheme.
  - storage-loc: Location to store data (TFRecords or occupancy maps).
  - prefix: String to prepend to TFRecord. (only available for `convert_kitti_large.py`)
  - samples-per-record: Number of sequences per TFRecord.
  - K: Number of frames to observe before prediction.
  - T: Number of frames to predict.
  - image-size: Size of grid map.
  - seq-length: How many frames per Sequence. Has to be at least K+T+1. Useful when testing different (K,T) combinations.
  - step-size: Number of frames to skip between sequences.

Run the script with
```
python convert_kitti_large.py --date="2011_09_11" --drive="0005" --storage-loc="./tfrecords/" --prefix="data" --samples-per-record=20 --K=9 --T=10 --image-size=96 --seq-length=20 --step-size=5
```
