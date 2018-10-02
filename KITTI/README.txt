kitti conversion scripts:

kitti_converter.py - converts KITTI data to odometry.txt and occupancy maps similar to raw data from MBRDNA

convert_kitti_large.py - converts KITTI data to TFRecord directly, includes grid map, RGB and black images for segmentation and depth.

NOTE before running: change basedir in convert_kitti_large.py/kitti_converter.py to where KITTI data is stored!
run using:
python convert_kitti_large.py

arguments are:
--date, --drive which are part of the KITTI naming scheme
--storage_loc where to put TFRecord/occupancy maps
--prefix to append before TFRecord (only for convert_kitti_large.py)
