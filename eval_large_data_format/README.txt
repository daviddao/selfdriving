running evaluation code:

run
  CUDA_VISIBLE_DEVICES=0 python -W ignore test_onmove_dataset.py --num_gpu=1
or to run with other provided tfRecord:
  CUDA_VISIBLE_DEVICES=0 python -W ignore test_onmove_dataset.py --num_gpu=1 --tfrecord="evaldata2_imgsze=96_fc=20_datasze=240x80_seqlen=1_K=9_T=10_size=20"
  (currently overwrites previous results. if both evals should be saved,
    please rename results folder before rerunning with other tfRecord.)

note: important arguments are:
  --data_path=<folder where tfRecords are stored>
  --tfrecord=<name of a tfRecord>

additional info:
  the model is given 9 (K) frames and predicts the next 10 (T) frames.
  the prediction is based on the previous predicted frame (not ground truth).
  (example: 16. frame is based on predicted 15. frame, not on 15. gt frame)

the results will be stored in ../results/
important part of the result will be in:
../results/images/Gridmap/<modelname>/saveX/img_X.png

The image is structured as follows:
  first row: first prediction (9. frame if (K=9,T=10))
  last row: last prediction (19. frame if (K=9,T=10))
  first column of a pair is always ground truth, other is prediction.

simplified model information:
-predicts gridmap through series of convolutions and LSTM layers, at the core
  is a VAE which encodes the convolved input as a multivariate normal distr.
  from which it samples and deconvolves back to the original input shape.
  training loss is computed with BCE from unoccluded parts of the gridmap.

-predicts RGB/Segmentation/Depth images by reducing image dimensionality through
  series of convolutions & LSTM layers. at reduced dimensionality concatenates
  images and appends transformed gridmap, tf_matrix (containing speed & yawrate)
  and direction (going straight/left/right). at the center sits again a VAE, its
  output is deconvolved back to the original shape and RGB/Seg/Depth images are
  extracted.
  training loss is computed from MSE of each image summed up.
