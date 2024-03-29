from carla.client import make_carla_client
from carla.sensor import Camera, Lidar
from carla.settings import CarlaSettings
from carla.tcp import TCPConnectionError
from carla.util import print_over_same_line

from PIL import Image

import math
import time
import argparse
import logging
import random
import os

#import gym #pip install gym
import numpy as np
from cntk.core import Value #pip install cntk-gpu
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Convolution3D, Dense, default_options
from cntk.layers.typing import Signature, Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, argmax, element_select, less, relu, reduce_max, reduce_sum, square
from cntk.ops.functions import CloneMethod, Function
from cntk.train import Trainer

import pickle

class ReplayMemory(object):
    """
    ReplayMemory keeps track of the environment dynamic.
    We store all the transitions (s(t), action, s(t+1), reward, done).
    The replay memory allows us to efficiently sample minibatches from it, and generate the correct state representation
    (w.r.t the number of previous frames needed).
    """
    def __init__(self, size, sample_shape, history_length=4):
        self._pos = 0
        self._count = 0
        self._max_size = size
        self._history_length = max(1, history_length)
        self._state_shape = sample_shape
        self._states = np.zeros((size,) + sample_shape, dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.uint8)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._terminals = np.zeros(size, dtype=np.float32)

    def __len__(self):
        """ Returns the number of items currently present in the memory
        Returns: Int >= 0
        """
        return self._count

    def append(self, state, action, reward, done):
        """ Appends the specified transition to the memory.

        Attributes:
            state (Tensor[sample_shape]): The state to append
            action (int): An integer representing the action done
            reward (float): An integer representing the reward received for doing this action
            done (bool): A boolean specifying if this state is a terminal (episode has finished)
        """
        assert state.shape == self._state_shape, \
            'Invalid state shape (required: %s, got: %s)' % (self._state_shape, state.shape)

        self._states[self._pos] = state
        self._actions[self._pos] = action
        self._rewards[self._pos] = reward
        self._terminals[self._pos] = done

        self._count = max(self._count, self._pos + 1)
        self._pos = (self._pos + 1) % self._max_size

    def sample(self, size):
        """ Generate size random integers mapping indices in the memory.
            The returned indices can be retrieved using #get_state().
            See the method #minibatch() if you want to retrieve samples directly.

        Attributes:
            size (int): The minibatch size

        Returns:
             Indexes of the sampled states ([int])
        """

        # Local variable access is faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            if index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        return indexes

    def minibatch(self, size):
        """ Generate a minibatch with the number of samples specified by the size parameter.

        Attributes:
            size (int): Minibatch size

        Returns:
            tuple: Tensor[minibatch_size, input_shape...], [int], [float], [bool]
        """
        indexes = self.sample(size)

        pre_states = np.array([self.get_state(index) for index in indexes], dtype=np.float32)
        post_states = np.array([self.get_state(index + 1) for index in indexes], dtype=np.float32)
        actions = self._actions[indexes]
        rewards = self._rewards[indexes]
        dones = self._terminals[indexes]

        return pre_states, actions, post_states, rewards, dones

    def get_state(self, index):
        """
        Return the specified state with the replay memory. A state consists of
        the last `history_length` perceptions.

        Attributes:
            index (int): State's index

        Returns:
            State at specified index (Tensor[history_length, input_shape...])
        """
        if self._count == 0:
            raise IndexError('Empty Memory')

        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent
    for evaluation
    """

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """ Underlying buffer with N previous states stacked along first axis

        Returns:
            Tensor[shape]
        """
        return self._buffer

    def append(self, state):
        """ Append state to the history

        Attributes:
            state (Tensor) : The state to append to the memory
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = state

    def reset(self):
        """ Reset the memory. Underlying buffer set all indexes to 0

        """
        self._buffer.fill(0)

class LinearEpsilonAnnealingExplorer(object):
    """
    Exploration policy using Linear Epsilon Greedy

    Attributes:
        start (float): start value
        end (float): end value
        steps (int): number of steps between start and end
    """

    def __init__(self, start, end, steps):
        self._start = start
        self._stop = end
        self._steps = steps

        self._step_size = (end - start) / steps

    def __call__(self, num_actions):
        """
        Select a random action out of `num_actions` possibilities.

        Attributes:
            num_actions (int): Number of actions available
        """
        return np.random.choice(num_actions)

    def _epsilon(self, step):
        """ Compute the epsilon parameter according to the specified step

        Attributes:
            step (int)
        """
        if step < 0:
            return self._start
        elif step > self._steps:
            return self._stop
        else:
            return self._step_size * step + self._start

    def is_exploring(self, step):
        """ Commodity method indicating if the agent should explore

        Attributes:
            step (int) : Current step

        Returns:
             bool : True if exploring, False otherwise
        """
        return np.random.rand() < self._epsilon(step)

def huber_loss(y, y_hat, delta):
    """ Compute the Huber Loss as part of the model graph

    Huber Loss is more robust to outliers. It is defined as:
     if |y - y_hat| < delta :
        0.5 * (y - y_hat)**2
    else :
        delta * |y - y_hat| - 0.5 * delta**2

    Attributes:
        y (Tensor[-1, 1]): Target value
        y_hat(Tensor[-1, 1]): Estimated value
        delta (float): Outliers threshold

    Returns:
        CNTK Graph Node
    """
    half_delta_squared = 0.5 * delta * delta
    error = y - y_hat
    abs_error = abs(error)

    less_than = 0.5 * square(error)
    more_than = (delta * abs_error) - half_delta_squared
    loss_per_sample = element_select(less(abs_error, delta), less_than, more_than)

    return reduce_sum(loss_per_sample, name='loss')

class DeepQAgent(object):
    """
    Implementation of Deep Q Neural Network agent like in:
        Nature 518. "Human-level control through deep reinforcement learning" (Mnih & al. 2015)
    """
    def __init__(self, input_shape, nb_actions,
                 gamma=0.99, explorer=LinearEpsilonAnnealingExplorer(1, 0.1, 1000000),
                 learning_rate=0.00025, momentum=0.95, minibatch_size=32,
                 memory_size=500000, train_after=200000, train_interval=4, target_update_interval=10000,
                 monitor=True):
        self.input_shape = input_shape
        self.nb_actions = nb_actions
        self.gamma = gamma

        self._train_after = train_after
        self._train_interval = train_interval
        self._target_update_interval = target_update_interval

        self._explorer = explorer
        self._minibatch_size = minibatch_size
        self._history = History(input_shape)
        self._memory = ReplayMemory(memory_size, input_shape[1:], 4)
        self._num_actions_taken = 0

        # Metrics accumulator
        self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

        # Action Value model (used by agent to interact with the environment)
        with default_options(activation=relu, init=he_uniform()):
            self._action_value_net = Sequential([
                Convolution3D((8, 8, 3), 64, strides=1),
                Convolution3D((4, 4, 4), 32, strides=2),
                Convolution3D((3, 3, 1), 32, strides=2),
                Dense(256, init=he_uniform(scale=0.01)),
                Dense(nb_actions, activation=None, init=he_uniform(scale=0.01))
            ])
        self._action_value_net.update_signature(Tensor[input_shape])

        # Target model used to compute the target Q-values in training, updated
        # less frequently for increased stability.
        self._target_net = self._action_value_net.clone(CloneMethod.freeze)

        # Function computing Q-values targets as part of the computation graph
        @Function
        @Signature(post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def compute_q_targets(post_states, rewards, terminals):
            return element_select(
                terminals,
                rewards,
                gamma * reduce_max(self._target_net(post_states), axis=0) + rewards,
            )

        # Define the loss, using Huber Loss (more robust to outliers)
        @Function
        @Signature(pre_states=Tensor[input_shape], actions=Tensor[nb_actions],
                   post_states=Tensor[input_shape], rewards=Tensor[()], terminals=Tensor[()])
        def criterion(pre_states, actions, post_states, rewards, terminals):
            # Compute the q_targets
            q_targets = compute_q_targets(post_states, rewards, terminals)

            # actions is a 1-hot encoding of the action done by the agent
            q_acted = reduce_sum(self._action_value_net(pre_states) * actions, axis=0)

            # Define training criterion as the Huber Loss function
            return huber_loss(q_targets, q_acted, 1.0)

        # Adam based SGD
        lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
        m_schedule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)
        l_sgd = adam(self._action_value_net.parameters, lr_schedule,
                     momentum=m_schedule, variance_momentum=vm_schedule)

        self._metrics_writer = TensorBoardProgressWriter(freq=1, log_dir='metrics', model=criterion) if monitor else None
        self._learner = l_sgd
        self._trainer = Trainer(criterion, (criterion, None), l_sgd, self._metrics_writer)

        #self._trainer.restore_from_checkpoint('models/oldmodels/model800000')

    def act(self, state):
        """ This allows the agent to select the next action to perform in regard of the current state of the environment.
        It follows the terminology used in the Nature paper.

        Attributes:
            state (Tensor[input_shape]): The current environment state

        Returns: Int >= 0 : Next action to do
        """
        # Append the state to the short term memory (ie. History)
        self._history.append(state)

        # If policy requires agent to explore, sample random action
        if self._explorer.is_exploring(self._num_actions_taken):
            action = self._explorer(self.nb_actions)
            #print("exploring with action "+str(action))
        else:
            # Use the network to output the best action
            env_with_history = self._history.value
            q_values = self._action_value_net.eval(
                # Append batch axis with only one sample to evaluate
                env_with_history.reshape((1,) + env_with_history.shape)
            )

            self._episode_q_means.append(np.mean(q_values))
            self._episode_q_stddev.append(np.std(q_values))

            # Return the value maximizing the expected reward
            action = q_values.argmax()
            #print("taking decisive action "+str(action))

        # Keep track of interval action counter
        self._num_actions_taken += 1
        return action

    def observe(self, old_state, action, reward, done):
        """ This allows the agent to observe the output of doing the action it selected through act() on the old_state

        Attributes:
            old_state (Tensor[input_shape]): Previous environment state
            action (int): Action done by the agent
            reward (float): Reward for doing this action in the old_state environment
            done (bool): Indicate if the action has terminated the environment
        """
        self._episode_rewards.append(reward)

        # If done, reset short term memory (ie. History)
        if done:
            # Plot the metrics through Tensorboard and reset buffers
            if self._metrics_writer is not None:
                self._plot_metrics()
            self._episode_rewards, self._episode_q_means, self._episode_q_stddev = [], [], []

            # Reset the short term memory
            self._history.reset()

        # Append to long term memory
        self._memory.append(old_state, action, reward, done)

    def train(self):
        """ This allows the agent to train itself to better understand the environment dynamics.
        The agent will compute the expected reward for the state(t+1)
        and update the expected reward at step t according to this.

        The target expectation is computed through the Target Network, which is a more stable version
        of the Action Value Network for increasing training stability.

        The Target Network is a frozen copy of the Action Value Network updated as regular intervals.
        """

        agent_step = self._num_actions_taken

        if agent_step >= self._train_after:
            if (agent_step % self._train_interval) == 0:
                pre_states, actions, post_states, rewards, terminals = self._memory.minibatch(self._minibatch_size)
                self._trainer.train_minibatch(
                    self._trainer.loss_function.argument_map(
                        pre_states=pre_states,
                        actions=Value.one_hot(actions.reshape(-1, 1).tolist(), self.nb_actions),
                        post_states=post_states,
                        rewards=rewards,
                        terminals=terminals
                    )
                )

                # Update the Target Network if needed
                if (agent_step % self._target_update_interval) == 0:
                    self._target_net = self._action_value_net.clone(CloneMethod.freeze)
                    filename = "models\model%d" % agent_step
                    self._trainer.save_checkpoint(filename)

    def _plot_metrics(self):
        """Plot current buffers accumulated values to visualize agent learning
        """
        if len(self._episode_q_means) > 0:
            mean_q = np.asscalar(np.mean(self._episode_q_means))
            self._metrics_writer.write_value('Mean Q per ep.', mean_q, self._num_actions_taken)

        if len(self._episode_q_stddev) > 0:
            std_q = np.asscalar(np.mean(self._episode_q_stddev))
            self._metrics_writer.write_value('Mean Std Q per ep.', std_q, self._num_actions_taken)

        self._metrics_writer.write_value('Sum rewards per ep.', sum(self._episode_rewards), self._num_actions_taken)


def transform_input(responses, convert=True):
    responses = Image.fromarray(np.uint8(responses))
    if convert:
        return np.array(responses.resize((120, 40), Image.LANCZOS).convert('L'))
    else:
        return np.array(responses.resize((120, 40), Image.LANCZOS))

def interpret_action(action, control):
    control.hand_brake = False
    control.reverse = False
    control.steer = 0
    control.brake = 0
    control.throttle = 1
    if action == 0:
        control.throttle = 0
        control.brake = 1
    elif action == 1:
        control.steer = 0
    elif action == 2:
        control.steer = 0.5
    elif action == 3:
        control.steer = -0.5
    elif action == 4:
        control.steer = 0.25
    else:
        control.steer = -0.25
    return control

def compute_reward(measurements, starting_position):
    player_measurements = measurements.player_measurements
    for agent in measurements.non_player_agents:
        agent.id # unique id of the agent
        if agent.HasField('speed_limit_sign'):
            MAX_SPEED = agent.speed_limit_sign.speed_limit

    MIN_SPEED = 1
    thresh_dist = 3.5
    beta = 3

    z = 0
    pts = [np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, 125, z]), np.array([0, 125, z]), np.array([0, -1, z]), np.array([130, -1, z]), np.array([130, -128, z]), np.array([0, -128, z]), np.array([0, -1, z])]
    car_pt = np.array([player_measurements.transform.location.x,
                       player_measurements.transform.location.y,
                       player_measurements.transform.location.z])

    dist = 10000000
    for i in range(0, len(pts)-1):
        dist = min(dist, np.linalg.norm(np.cross((car_pt - pts[i]), (car_pt - pts[i+1])))/np.linalg.norm(pts[i]-pts[i+1]))

    reward_dist = (math.exp(-beta*dist) - 0.5)
    reward_speed = (((player_measurements.forward_speed - MIN_SPEED)/(MAX_SPEED - MIN_SPEED)) - 0.5)
    reward = reward_dist + reward_speed

    print("Distance travelled="+str(dist)+". Reward Dist="+str(reward_dist)+". Reward Speed="+str(reward_speed)+".")
    return reward

def isDone(measurements, control, reward):
    done = 0
    player_measurements = measurements.player_measurements
    if player_measurements.collision_vehicles > 0 or player_measurements.collision_pedestrians > 0 or player_measurements.collision_other > 0 or player_measurements.intersection_otherlane > 0.0 or player_measurements.intersection_offroad > 0.0:
        done = 1

    return done

def run_carla_client(args):
    with make_carla_client(args.host, args.port, timeout=500) as client:
        print('CarlaClient connected')

        if args.settings_filepath is None:

            settings = CarlaSettings()
            settings.set(
                SynchronousMode=args.synchronous_mode,
                SendNonPlayerAgentsInfo=True,
                NumberOfVehicles=0,
                NumberOfPedestrians=0,
                WeatherId=random.choice([1, 3, 7, 8, 14]),
                QualityLevel=args.quality_level)
            settings.randomize_seeds()

            camera0 = Camera('CameraRGB')
            camera0.set_image_size(1920, 640)
            camera0.set_position(2.00, 0, 1.30)
            settings.add_sensor(camera0)

            camera1 = Camera('CameraDepth', PostProcessing='Depth')
            camera1.set_image_size(1920, 640)
            camera1.set_position(2.00, 0, 1.30)
            settings.add_sensor(camera1)

            camera2 = Camera('CameraSegmentation', PostProcessing='SemanticSegmentation')
            camera2.set_image_size(1920, 640)
            camera2.set_position(2.00, 0, 1.30)
            settings.add_sensor(camera2)

        else:

            with open(args.settings_filepath, 'r') as fp:
                settings = fp.read()

        scene = client.load_settings(settings)

        number_of_player_starts = len(scene.player_start_spots)
        #player_start = random.randint(0, max(0, number_of_player_starts - 1))

        print('Starting new episode at %r...' % scene.map_name)
        client.start_episode(0)
        measurements, sensor_data = client.read_data()
        #only works because we always start at the same spot
        starting_position = [measurements.player_measurements.transform.location.x,
                             measurements.player_measurements.transform.location.y,
                             measurements.player_measurements.transform.location.z]

        # Make RL agent
        NumBufferFrames = 4
        SizeRows = 40
        SizeCols = 120
        NumActions = 6
        agent = DeepQAgent((NumBufferFrames, SizeRows, SizeCols, 7), NumActions, monitor=True, memory_size=50000)

        # Train
        epoch = 100
        current_step = 0
        max_steps = epoch * 250000

        measurements, sensor_data = client.read_data()
        for name, measurement in sensor_data.items():
            if name == 'CameraDepth':
                depth = measurement.return_depth_map()
            if name == 'CameraSegmentation':
                segmentation = measurement.return_segmentation_map()
            if name == 'CameraRGB':
                rgb = measurement.return_rgb()

        current_state = np.concatenate([np.expand_dims(transform_input(depth),2), transform_input(segmentation, False), transform_input(rgb, False)],axis=2)

        while True:
            action = agent.act(current_state)
            #obtain control handler
            control = measurements.player_measurements.autopilot_control
            control = interpret_action(action, control)
            client.send_control(control)

            measurements, sensor_data = client.read_data()
            reward = compute_reward(measurements, starting_position)
            done = isDone(measurements, control, reward)

            if done == 1:
                reward = -10

            agent.observe(current_state, action, reward, done)
            agent.train()

            if done:
                #restart at same place for now
                client.start_episode(0)
                #obtain control handler
                control = measurements.player_measurements.autopilot_control
                control = interpret_action(action, control)
                client.send_control(control)
                time.sleep(1)
                current_step +=1

            for name, measurement in sensor_data.items():
                if name == 'CameraDepth':
                    depth = measurement.return_depth_map()
                if name == 'CameraSegmentation':
                    segmentation = measurement.return_segmentation_map()
                if name == 'CameraRGB':
                    rgb = measurement.return_rgb()

            current_state = np.concatenate([np.expand_dims(transform_input(depth),2), transform_input(segmentation, False), transform_input(rgb, False)],axis=2)


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host server (default: localhost)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-q', '--quality-level',
        choices=['Low', 'Epic'],
        type=lambda s: s.title(),
        default='Epic',
        help='graphics quality level, a lower level makes the simulation run considerably faster.')
    argparser.add_argument(
        '-c', '--carla-settings',
        metavar='PATH',
        dest='settings_filepath',
        default=None,
        help='Path to a "CarlaSettings.ini" file')
    argparser.add_argument(
        '-s', '--synchronous-mode',
        dest='synchronous_mode',
        default=True,
        help='Synchronous or Asynchronous mode?')

    args = argparser.parse_args()

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    exp_backoff = 1
    while True:
        try:

            run_carla_client(args)

            print('Done.')
            return

        except TCPConnectionError as error:
            logging.error(error)
            time.sleep(exp_backoff)
            exp_backoff *= 2


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
