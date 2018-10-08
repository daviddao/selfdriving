how to run preprocessing code:

run convert_dataset_new.sh

when asked which dataset to convert, type 1 (or other random value)

it will say "dataset unknown, please enter the source directory of your frames"
  enter entire path to dataset (where json and images are located)
  (ex. /mnt/ds3lab-scratch/lucala/20180117_1555_cw_big_loop_sunnyvale_ave_traffic_lights_annotation/ )

now it will ask to name the new dataset, choose a unique name (such as BigLoop1555 in the above case)


The preprocessing script will work through the folder, it may throw errors but these can be ignored!
You will find the converted dataset at ./preprocessed_dataset/
  where the final file will be a tfRecord which can be fed into the test script
