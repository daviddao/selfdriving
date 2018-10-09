# Training documentation
A short documentation for network training.

## Model information
The training script preloads the TFRecords using the Data API from Tensorflow. In case of Multi-GPU it distributes samples to the GPUs and collects the gradients of every GPU (called tower) and backpropagates the weight updates. The loss is computed separately for the grid map portion of the network and the image model. The grid map loss consists of a binary cross-entropy loss with a KL divergence term. The image model loss consists of a binary cross-entropy loss, a KL divergence term and an odometry term. The odometry loss encapsulates the error of the direction vector, transformation matrix and the speed and yaw rate. For more information see the thesis report.

Every `img-freq` interval the model outputs a text file containing the ground truth and predicted speed and yaw rate, an image containing the ground truth grid map, predicted grid map and intermediate grid maps in `train_x_0.png` as well as the ground truth and predicted image channels in `train_cam_x_9.png`.

The important files are `train_onmove_distr_dataset.py` which handles the training of the network defined in `move_network_distributed_noGPU.py`. The network uses convolutional LSTM cells defined in `BasicConvLSTMCell.py` and a Spatial Transformer Module defined in `spatial_transformer.py` as well as SELU activation and diluted convolutions defined in `additional_ops.py` and deconvolutional operations defined in `ops.py`. The parsing of the TFRecord filename is done in `utils.py`.

## Running training scripts
Important arguments that can be passed to the script are:
  - CUDA_VISIBLE_DEVICES: Needed for the system to be able to observe the GPUs. Tensorflow will allocate resources to all visible GPUs even if they are not used with the `--gpu` flag.
  - gpu: Array of GPUs to use. Cannot be larger than `CUDA_VISIBLE_DEVICES` and has to start at 0.
  - prefix: String prepended to the model.
  - num-iter: Number of iterations to train.
  - img-freq: Iteration interval between saving model and image output.
  - data-path-scratch: Location where TFRecords are stored.
  - tfrecord: Name of the TFRecord to use. Can be left empty if all TFRecords inside `data-path-scratch` should be used, in this case a random TFRecord filename will be parsed for the necessary information.

These are the most important arguments to get started. More arguments to control the learning process can be can be found at the end of `train_onmove_distr_dataset.py`.

Use the following command to run the training script on 4 GPUs:
```
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_onmove_distr_dataset.py --gpu 0 1 2 3 --dense-block=False --prefix="pure-speedyaw_" --num-iter=50000 --img-freq=250 --
data-path-scratch="/mnt/ds3lab-scratch/lucala/datasets/CARLA/" --tfrecord=""
```

The above case has 8 GPUs, where the last four are being used. Adjust accordingly.

Note: Many features from the GAN network are still present for backward compatibility. However, they have not been tested and are not meant to be used for the current version of the code.
