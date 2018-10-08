how to run evaluation code:

unzip model located in ./models (this model was trained for 88k iterations on old data)

then run following code:
CUDA_VISIBLE_DEVICES=0 python test_onmove_dataset.py --num_gpu=1

interesting flags:
--road        include road in final png/gif
--num_iters   how many files to evaluate the tfrecord
