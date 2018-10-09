# Evaluation documentation
A short documentation for network evaluation. Two compressed TFRecords are included in the `/tfrecords` subdirectory. They need to be decompressed before use. A zipped model checkpoint is given in `/model` and can be used to run this code out of the box. For more model checkpoint files and datasets see `/mnt/ds3lab-scratch/lucala`.

## Running evaluation scripts
Start by unzipping the model stored in `/model` and the TFRecords stored in `/tfrecords`.

Important arguments that can be passed to the script are:
  - CUDA_VISIBLE_DEVICES: Needed for the system to be able to observe the GPUs. Tensorflow will allocate resources to all visible GPUs.
  - num-gpu: Number of GPUs to use. Cannot be larger than `CUDA_VISIBLE_DEVICES`.
  - num-iter: Number of iterations to loop over TFRecord. Should be equal to the number of sequences contained in a TFRecord.
  - denseBlock: Train using the dense block (network of diluted convolutions) or VAE. This needs to be the same as the model. Cannot evaluate a VAE on a model that trained a dense block.
  - data-path: Location where TFRecords are stored.
  - tfrecord: Name of the TFRecord to use.
  - chckpt-loc: Location of the model checkpoint file.
  - speed-yaw-loss: Output additional text file containing ground truth and predicted odometry (speed, yaw rate).

Use the following command to run the evaluation script:
```
CUDA_VISIBLE_DEVICES=0 python -W ignore test_onmove_dataset.py --num-gpu=1
```

The evaluation script supports Multi-GPU but does not need it. Running with 1 GPU is enough for evaluation.
