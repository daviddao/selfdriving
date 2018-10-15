# Evaluation documentation
A short documentation for network evaluation. Two compressed TFRecords are included in the `/tfrecords` subdirectory. They need to be decompressed before use. A zipped model checkpoint is given in `/model` and can be used to run this code out of the box. For more model checkpoint files and datasets see `/mnt/ds3lab-scratch/lucala`.

## Running evaluation scripts
Start by unzipping the model stored in `/model` and the TFRecords stored in `/tfrecords`.

Important arguments that can be passed to the script are:
  - CUDA_VISIBLE_DEVICES: Needed for the system to be able to observe the GPUs. Tensorflow will allocate resources to all visible GPUs.
  - num-gpu: Number of GPUs to use. Cannot be larger than `CUDA_VISIBLE_DEVICES`.
  - num-iter: Number of iterations to loop over TFRecord. Should be equal to the number of sequences contained in a TFRecord.
  - denseBlock: Evaluate using the dense block (network of diluted convolutions) or VAE. This needs to be the same as the model. Cannot evaluate a VAE on a model that trained a dense block.
  - data-path: Location where TFRecords are stored.
  - tfrecord: Name of the TFRecord to use.
  - chckpt-loc: Location of the model checkpoint file.
  - speed-yaw-loss: Output additional text file containing ground truth and predicted odometry (speed, yaw rate).
  - prefix: Name of the model to use.
  - cutout_input: Cut out a certain channel of the input to see influence of other channels. 0 - default, 1 - remove grid map, 2 - remove RGB, 3 - remove segmentation, 4 - remove depth. Prediction ground truth still visible for comparison.

Use the following command to run the evaluation script with the provided TFRecord and model:
```
CUDA_VISIBLE_DEVICES=0 python -W ignore test_onmove_dataset.py
```
Note: The evaluation script needs 1 GPU to run.

The results will be stored in `../results` where the prediction for every sequence `X` is saved under `/results/images/Gridmap/<model name>/saveX/`. The `gt_*.png` are grid map ground truth frames that are concatenated to a gif in `gt_X.gif`. The same is true for the predictions `pred_*.png`. The images `img_gt_X.gif` and `img_pred_X.gif` contain the ground truth and predicted image channel frame sequence. They both show all (K+T) frames. The image `img_X.png` shows the `T` future frame predictions of all channels side by side with the ground truth always being on the left. The topmost row is the first prediction and the bottom row is the last prediction. If the ground truth of an image channel is black it is because it is not provided in that dataset.
