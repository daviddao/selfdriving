run training using train_onmove_distr_dataset.py

CUDA_VISIBLE_DEVICES=7 python train_onmove_distr_dataset.py --gpu 0 --prefix="pure-test_" --num_iter=100000 --imgFreq=500
--data_path_scratch="./" --tfrecord="2_0_0_imgsze=96_fc=20_datasze=240x80_seqlen=1_K=9_T=10_size=20"

arguments:
CUDA_VISIBLE_DEVICES number of GPUs to use for training

--gpu 0 1 2 - in case of 3 GPUs, always needs to start with 0, needs to be same amount as CUDA_VISIBLE_DEVICES

--prefix - the name prepended to the model

--num_iter - number of iterations to train

--imgFreq - how often to save images (every 500. iteration in above ex.)

--data_path_scratch - where tfRecords are located

--tfrecord - name of tfrecord
