"""
Taken from tensorflow tutorial and modified
https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/cifar10_main.py
"""



from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os

from move_network_estimator import MCNET
import cifar10
import cifar10_utils
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)


def get_model_fn(num_gpus, variable_strategy, num_workers):
    """Returns a function that will build the resnet model."""

    def _movenet_model_fn(features, labels, mode, params):
        """Resnet model body.

        Support single host, one or more GPU training. Parameter distribution can
        be either one of the following scheme.
        1. CPU is the parameter server and manages gradient updates.
        2. Parameters are distributed evenly across all GPUs, and the first GPU
           manages gradient updates.

        Args:
          features: a list of tensors, one for each tower
          labels: a list of tensors, one for each tower
          mode: ModeKeys.TRAIN or EVAL
          params: Hyperparameters suitable for tuning
        Returns:
          A EstimatorSpec object.
        """
        
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        lr_D = params.lr_D
        lr_G = params.lr_G
        beta = params.beta

        tower_features = features
        tower_losses = []
        tower_gradvars = []
        tower_preds = []

        # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
        # on CPU. The exception is Intel MKL on CPU which is optimal with
        # channels_last.
        data_format = params.data_format
        if not data_format:
            if num_gpus == 0:
                data_format = 'channels_last'
            else:
                data_format = 'channels_first'

        if num_gpus == 0:
            num_devices = 1
            device_type = 'cpu'
        else:
            num_devices = num_gpus
            device_type = 'gpu'

        for i in range(num_devices):
            worker_device = '/{}:{}'.format(device_type, i)
            if variable_strategy == 'CPU':
                device_setter = cifar10_utils.local_device_setter(
                    worker_device=worker_device)
            elif variable_strategy == 'GPU':
                device_setter = cifar10_utils.local_device_setter(
                    ps_device_type='gpu',
                    worker_device=worker_device,
                    ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                        num_gpus, tf.contrib.training.byte_size_load_fn))
            with tf.variable_scope('movenet', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = _tower_fn(
                            is_training, params.image_size, params.K, params.T, 
                            params.sequence_steps, params.d_input_frames, params.predOcclValue, params.alpha, beta, 
                            params.useSharpen, tower_features[i], data_format, i)
                        #might be wrong, only gives one sample per tower
                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                          # Only trigger batch_norm moving mean and variance update from
                          # the 1st tower. Ideally, we should grab the updates from all
                          # towers but these stats accumulate extremely fast so we can
                          # ignore the other stats from the other towers without
                          # significant detriment.
                          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                         name_scope)

        # Now compute global loss and gradients.
        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
                gradvars.append((avg_grad, var))

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):
            # Suggested learning rate scheduling from
            # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
            num_batches_per_epoch = cifar10.Cifar10DataSet.num_examples_per_epoch(
              'train') // (params.train_batch_size * num_workers)
            #boundaries = [
            #  num_batches_per_epoch * x
            #  for x in np.array([82, 123, 300], dtype=np.int64)
            #]
            #staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]]

            #learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
            #                                          boundaries, staged_lr)

            loss = tf.reduce_mean(tower_losses, name='loss')

            examples_sec_hook = cifar10_utils.ExamplesPerSecondHook(
                params.train_batch_size, every_n_steps=10)

            #tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}
            tensors_to_log = {'loss': loss}

            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=100)

            train_hooks = [logging_hook, examples_sec_hook]

            if beta != 0: #opt_D not being used ATM!
                opt_D = tf.train.AdamOptimizer(lr_D, beta1=0.5)

            optimizer = tf.train.AdamOptimizer(lr_G, beta1=0.5)

            if params.sync:
                optimizer = tf.train.SyncReplicasOptimizer(
                    optimizer, replicas_to_aggregate=num_workers)
                sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
                train_hooks.append(sync_replicas_hook)

            # Create single grouped train op
            train_op = [
              optimizer.apply_gradients(
                  gradvars, global_step=tf.train.get_global_step())
            ]
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)
            
            predictions = {
                'pred':
                    tf.concat([p['pred'] for p in tower_preds], axis=0),
                'gt_masked':
                    tf.concat([p['gt_masked'] for p in tower_preds], axis=0),
                'pred_masked':
                    tf.concat([p['pred_masked'] for p in tower_preds], axis=0),
                'trans_pred':
                    tf.concat([p['trans_pred'] for p in tower_preds], axis=0),
                'pre_trans_pred':
                    tf.concat([p['pre_trans_pred'] for p in tower_preds], axis=0)
            }
            
            #original = tf.concat([f for f in features[0]], axis=0)
            #print(predictions['gt_masked'].shape)
            #print(predictions['pred_masked'].shape)
            #print(tf.reduce_max(predictions['gt_masked']))
            #print(tf.reduce_max(predictions['pred_masked']))
            #print(tf.reduce_min(predictions['gt_masked']))
            #print(tf.reduce_min(predictions['pred_masked']))
            metrics = {
              'avg_psnr':
                  tf.metrics.mean(cifar10_utils.psnr(predictions['gt_masked'], predictions['pred_masked'], max_val=1.0))
            }

        return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=predictions,
                loss=loss,
                train_op=train_op,
                training_hooks=train_hooks,
                eval_metric_ops=metrics)

    return _movenet_model_fn


def _tower_fn(is_training, image_size, K, T, sequence_steps, d_input_frames, predOcclValue, alpha, beta, 
              useSharpen, features, data_format, i):
    """Build computation tower (Resnet).

    Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.

    Returns:
    A tuple with the loss for the tower, the gradients and parameters, and
    predictions.

    """

    seq_batch = features[0]
    input_batch = features[1]
    map_batch = features[2]
    transformation_batch = features[3]
    batch_size = int(seq_batch.shape[0])
    print("BATCH_SIZE_PER_GPU="+str(batch_size))
    
    model = MCNET(image_size=[image_size, image_size], c_dim=1,
          K=K, batch_size=batch_size, T=T,
          #checkpoint_dir=checkpoint_dir,
          iterations=sequence_steps,
          d_input_frames=d_input_frames,
          useSELU=True, motion_map_dims=2,
          showFutureMaps=True,
          predOcclValue=predOcclValue,
          useGAN=(beta != 0), useSharpen=useSharpen, data_format=data_format) #gpu dummy value
    #logits = model.forward_pass(feature, input_data_format='channels_last')
    pred, trans_pred, pre_trans_pred = model.forward(input_batch, map_batch, transformation_batch, i, input_data_format='channels_last')
    # Calculate the loss for this tower   
    model.target = seq_batch
    model.motion_map_tensor = map_batch
    model.G = tf.stack(axis=3, values=pred)
    model.G_trans = tf.stack(axis=3, values=trans_pred)
    model.G_before_trans = tf.stack(axis=3, values=pre_trans_pred)
    model.loss_occlusion_mask = (tf.tile(seq_batch[:, :, :, :, -1:], [1, 1, 1, 1, model.c_dim]) + 1) / 2.0
    model.target_masked = model.mask_black(seq_batch[:, :, :, :, :model.c_dim], model.loss_occlusion_mask)
    model.G_masked = model.mask_black(model.G, model.loss_occlusion_mask)
                            
    if beta != 0:
        start_frame = 0
        center_frame = model.d_input_frames // 2
        end_frame = model.d_input_frames
        gen_sequence = tf.concat(axis=3, values=[model.target_masked[
                                 :, :, :, start_frame:center_frame, :], model.G_masked[:, :, :, center_frame:end_frame, :]])
        gt_sequence = model.target_masked[:, :, :, start_frame:end_frame, :]
        good_data = tf.reshape(gt_sequence,
                               [model.batch_size, model.image_size[0],
                                model.image_size[1], -1])
        gen_data = tf.reshape(gen_sequence,
                              [model.batch_size, model.image_size[0],
                               model.image_size[1], -1])

        with tf.variable_scope("DIS", reuse=False):
            model.D, model.D_logits = model.discriminator(good_data)

        with tf.variable_scope("DIS", reuse=True):
            model.D_, model.D_logits_ = model.discriminator(gen_data)

        # Standard loss for real and fake (only for display and parameter
        # purpose, no loss trained on)

        model.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=model.D_logits, labels=tf.ones_like(model.D)
            )
        )
        model.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=model.D_logits_, labels=tf.zeros_like(model.D_)
            )
        )


    # specify loss to parameters
    model_params = tf.trainable_variables()
    #model.t_vars = model_params
    #model.g_vars = [var for var in model.t_vars if 'DIS' not in var.name]
    #if beta != 0:
    #    model.d_vars = [var for var in model.t_vars if 'DIS' in var.name]

    # Calculate the losses specific to encoder, generator, decoder                         
    model.L_BCE = model.weighted_BCE_loss(model.G_masked, model.target_masked) #cross-entropy mean
    #model.L_BCE = model.L_img
    if (beta != 0): #use GAN
        model.L_GAN = -tf.reduce_mean(model.D_)
        model.d_loss = model.d_loss_fake + model.d_loss_real
    else:
        model.d_loss_fake = tf.constant(0.0)
        model.d_loss_real = tf.constant(0.0)
        model.d_loss = tf.constant(0.0)
        model.L_GAN = tf.constant(0.0)

    model.L_p = tf.reduce_mean(tf.square(model.G_masked - model.target_masked))

    tower_loss = model.L_BCE

    #model_params = tf.trainable_variables()
    #tower_loss += weight_decay * tf.add_n(
    #  [tf.nn.l2_loss(v) for v in model_params])

    #tower_grad = tf.gradients(tower_loss, model_params)
    tower_grad = tf.gradients(alpha * tower_loss + beta * model.L_GAN, model_params)
    
    tower_pred = {
        'pred': tf.squeeze(model.G),
        'gt_masked': tf.squeeze(model.target_masked),
        'pred_masked': tf.squeeze(model.G_masked),
        'trans_pred': tf.squeeze(model.G_trans),
        'pre_trans_pred': tf.squeeze(model.G_before_trans)
    }

    return tower_loss, zip(tower_grad, model_params), tower_pred


def input_fn(data_dir,
             subset,
             num_shards,
             params,
             batch_size):
    """Create input graph for model.

    Args:
        data_dir: Directory where TFRecords representing the dataset are located.
        subset: one of 'train', 'validate' and 'eval'.
        num_shards: num of towers participating in data-parallel training.
        batch_size: total batch size for training to be divided by the number of
        shards.
        use_distortion_for_training: True to use distortions.
    Returns:
        two lists of tensors for features and labels, each of num_shards length.
    """
    with tf.device('/cpu:0'):
        dataset = cifar10.Cifar10DataSet(data_dir, subset, params.K, params.T, params.sequence_steps, params.tfrecordname_train, params.tfrecordname_eval)
        seq_batch, input_batch, map_batch, transformation_batch = dataset.make_batch(batch_size)

        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [[seq_batch, input_batch, map_batch, transformation_batch]], []

        # Note that passing num=batch_size is safe here, even though
        # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
        # examples. This is because it does so only when repeating for a limited
        # number of epochs, but our dataset repeats forever.


        seq_batch = tf.unstack(seq_batch, num=batch_size, axis=0)
        input_batch = tf.unstack(input_batch, num=batch_size, axis=0)
        map_batch = tf.unstack(map_batch, num=batch_size, axis=0)
        transformation_batch = tf.unstack(transformation_batch, num=batch_size, axis=0)
        feature_shards_0 = [[] for i in range(num_shards)]
        feature_shards_1 = [[] for i in range(num_shards)]
        feature_shards_2 = [[] for i in range(num_shards)]
        feature_shards_3 = [[] for i in range(num_shards)]

        for i in xrange(batch_size):
            idx = i % num_shards
            feature_shards_0[idx].append(seq_batch[i])
            feature_shards_1[idx].append(input_batch[i])
            feature_shards_2[idx].append(map_batch[i])
            feature_shards_3[idx].append(transformation_batch[i])

        feature_shards_0 = [tf.parallel_stack(x) for x in feature_shards_0]
        feature_shards_1 = [tf.parallel_stack(x) for x in feature_shards_1]
        feature_shards_2 = [tf.parallel_stack(x) for x in feature_shards_2]
        feature_shards_3 = [tf.parallel_stack(x) for x in feature_shards_3]
        #label_shards = [tf.parallel_stack(x) for x in label_shards]
        feature_shards = []
        for i in range(num_shards):
            feature_shards.append([feature_shards_0[i], feature_shards_1[i], feature_shards_2[i], feature_shards_3[i]])

        return feature_shards, [] #labels are dummy values. entire (K+T)*seq_len sequence is in feature_shards


def get_experiment_fn(data_dir,
                      num_gpus,
                      run_config,
                      hparams,
                      job_dir,
                      variable_strategy):
    """Returns an Experiment function.

    Experiments perform training on several workers in parallel,
    in other words experiments know how to invoke train and eval in a sensible
    fashion for distributed training. Arguments passed directly to this
    function are not tunable, all other arguments should be passed within
    tf.HParams, passed to the enclosed function.

    Args:
      data_dir: str. Location of the data for input_fns.
      num_gpus: int. Number of GPUs on each worker.
      variable_strategy: String. CPU to use CPU as the parameter server
      and GPU to use the GPUs as the parameter server.
    Returns:
      A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
      tf.contrib.learn.Experiment.

      Suitable for use by tf.contrib.learn.learn_runner, which will run various
      methods on Experiment (train, evaluate) based on information
      about the current runner in `run_config`.
    """

    #def _experiment_fn(run_config, hparams):
    """Returns an Experiment."""
    # Create estimator.
    train_input_fn = functools.partial(
        input_fn,
        data_dir,
        subset='train',
        num_shards=num_gpus,
        params=hparams,
        batch_size=hparams.train_batch_size)

    eval_input_fn = functools.partial(
        input_fn,
        data_dir,
        subset='eval',
        batch_size=hparams.eval_batch_size,
        params=hparams,
        num_shards=num_gpus)

    num_eval_examples = cifar10.Cifar10DataSet.num_examples_per_epoch('eval')
    if num_eval_examples % hparams.eval_batch_size != 0:
        raise ValueError(
            'validation set size must be multiple of eval_batch_size')

    #train_steps = hparams.train_steps
    eval_steps = num_eval_examples // hparams.eval_batch_size

    classifier = tf.estimator.Estimator(
        model_fn=get_model_fn(num_gpus, variable_strategy,
                          run_config.num_worker_replicas or 1),
        model_dir=job_dir,
        config=run_config,
        params=hparams)

    # Create experiment.
    #return tf.contrib.learn.Experiment(
    #    classifier,
    #    train_input_fn=train_input_fn,
    #    eval_input_fn=eval_input_fn,
    #    train_steps=train_steps,
    #    eval_steps=eval_steps)

    #return _experiment_fn
    return train_input_fn, eval_input_fn, classifier

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main(job_dir, data_dir, num_gpus, variable_strategy,
         log_device_placement, num_intra_threads,
         **hparams):
    # The env variable is on deprecation path, default is set to off.
    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    # Session configuration.
    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True))

    config = cifar10_utils.RunConfig(
      session_config=sess_config, model_dir=job_dir)
    kwargs = {'save_checkpoints_steps': 2500}
    config = config.replace(**kwargs)
    #other deprecated
    #estimator = tf.estimator.Estimator(model_fn=get_experiment_fn(data_dir, num_gpus, variable_strategy),
    #                                   config=config,
    #                                   params=tf.contrib.training.HParams(
    #                                      is_chief=config.is_chief,
    #                                      **hparams))
    params=tf.contrib.training.HParams(is_chief=config.is_chief,**hparams)

    train_input_fn, eval_input_fn, classifier = get_experiment_fn(data_dir, num_gpus, config, params, job_dir, variable_strategy)
    
    #train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=params.train_steps)
    #eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=100, throttle_secs = 12000)
    
    #tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    
    #tf.contrib.learn.learn_runner.run(
    #    get_experiment_fn(data_dir, num_gpus, variable_strategy),
    #    run_config=config,
    #    hparams=tf.contrib.training.HParams(
    #      is_chief=config.is_chief,
    #      **hparams))
    
    
    #might try this?
    for i in range(params.train_steps):
        classifier.train(input_fn=train_input_fn, steps=6000)# should be subset of training dataset
        classifier.evaluate(input_fn=eval_input_fn, steps=500)
    #    output = classifier.predict #create predict input fn with only a couple of samples
    #    #draw output using old code

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='The directory where the input data is stored.')
    parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.')
    parser.add_argument(
      '--variable-strategy',
      choices=['CPU', 'GPU'],
      type=str,
      default='CPU',
      help='Where to locate variable operations')
    parser.add_argument(
      '--num-gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.')
    parser.add_argument(
      '--train-steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.')
    parser.add_argument(
      '--train-batch-size',
      type=int,
      default=4,
      help='Batch size for training.')
    parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=1,
      help='Batch size for validation.')
    parser.add_argument(
      '--sync',
      action='store_true',
      default=False,
      help="""\
      If present when running in a distributed environment will run on sync mode.\
      """)
    parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
    parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """)
    parser.add_argument(
      '--data-format',
      type=str,
      default=None,
      help="""\
      If not set, the data format best for the training device is used. 
      Allowed values: channels_first (NCHW) channels_last (NHWC).\
      """)
    parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.')
    parser.add_argument("--lr_D", type=float, dest="lr_D",
                        default=0.0001, help="Base Learning Rate for Discriminator")
    parser.add_argument("--lr_G", type=float, dest="lr_G",
                        default=0.0001, help="Base Learning Rate for Generator")
    parser.add_argument("--alpha", type=float,
                        default=1.001, help="Image loss weight")
    parser.add_argument("--beta", type=float,
                        default=0.0, help="GAN loss weight")
    parser.add_argument("--image_size", type=int,
                        default=96, help="Training image size")
    parser.add_argument("--K", type=int,
                        default=9, help="Number of steps to observe from the past")
    parser.add_argument("--T", type=int,
                        default=10, help="Number of steps into the future")
    parser.add_argument("--sequence_steps", type=int,
                        default=4, help="Number of iterations per Sequence (K | T | K | T | ...) - one K + T step is one iteration")
    parser.add_argument("--d_input_frames", type=int,
                        default=20, help="How many frames the discriminator should get. Has to be at least K+T")
    parser.add_argument("--useSELU", type=str2bool,
                        default=True, help="If SELU should be used instead of RELU")
    parser.add_argument("--useCombinedMask", type=str2bool,
                        default=False, help="If SELU should be used instead of RELU")
    parser.add_argument("--predOcclValue", type=int,
                        default=-1, help="If SELU should be used instead of RELU")
    parser.add_argument("--imgFreq", type=int,
                        default=100, help="If SELU should be used instead of RELU")
    parser.add_argument("--prefix", type=str,
                        default="", help="Prefix appended to model name for easier search")
    parser.add_argument("--useSharpen", type=str2bool,
                        default=False, help="If sharpening should be used")
    parser.add_argument("--tfrecordname_train",
                        default="BigLoop2-5_Shard_imgsze=96_seqlen=4_K=10_T=14_all", help="tfrecord_train name")
    parser.add_argument("--tfrecordname_eval",
                        default="BigLoop1_imgsze=96_seqlen=4_K=10_T=20_all_in_one_all", help="tfrecord_eval name")

    args = parser.parse_args()

    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.num_gpus < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                     '--variable-strategy=CPU.')
    if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

    main(**vars(args))