# Documentation
A short documentation of dependencies, files and structure of the code for the master thesis "Deep Visual Foresight for Self-driving Cars". Additional documentation can be found in READMEs of subfolders and specific code files.

## Dependencies
All code is written in Python 3 (v3.6.5). Python packages used are:

  - tensorflow 1.8.0
  - tensorflow-gpu 1.8.0
  - tqdm 4.23.0
  - tk 8.6.7
  - scipy 1.1.0
  - scikit-learn 0.19.1
  - pillow 5.1.0
  - numpy 1.14.5
  - matplotlib 2.2.2
  - cudnn 7.1.4
  - cudatoolkit 9.0
  - glob2 0.6

## Data
Datasets for the project are:

  - MBRDNA
  - KITTI
  - CARLA

They can be found at
    /mnt/ds3lab-scratch/lucala/datasets/

Note: use the check_tfrecord jupyter notebook in the dataset directory to understand data format.

Every TFRecord file has 20 sequences and each sequence contains 20 frames divided into K=9, T=10. Every frame consists of the ground truth grid map, input grid map, road map, transformation matrix, front facing RGB, segmentation and depth image as well as the direction vector.

### MBRDNA
This dataset contains real-world data from Mercedes-Benz driving through Sunnyvale, California. Contains 1220 TFRecord files.

### KITTI
This dataset contains real-world data from the Karlsruhe Institute of Technology vehicle driving through Karlsruhe, Germany. Contains 19 TFRecord files.

### CARLA
This dataset contains virtually generated data from the CARLA autonomous driving simulator. Contains 1995 TFRecord files captured in Town01.

## Structure
The files are located in various folders. The model and data are stored on the `/mnt/ds3lab-scratch/lucala/` drive. The large data format refers to the data that has grid map information *and* front facing camera images.

### Folders
 - **carla**: Contains scripts to gather data while running the CARLA simulator.
 - **large data format**: Contains training and evaluation code for the large format.
 - **KITTI**: Contains conversion and helper scripts to convert KITTI data to our TFRecord format.
 - **MB data preprocessing**: Contains conversion and helper scripts to convert MBRDNA data to our TFRecord format.
 - **documents**: Contains documents from the original code by Phillip Lippe.
 - **storage**: Contains a variety of old and modified versions of the original code and our extended version.

 Consult READMEs in the subfolders for more details.
