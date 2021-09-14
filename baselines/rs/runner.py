import sys
sys.path.append('..')

import os
import pickle
import time
import collections
import copy

import numpy as np
import numpy.random as npr
import ray
import torch
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from models.state_prediction_model import StatePredictionModel
from data_generator import DataGenerator, ExpertDataGenerator
from data.replay_buffer import ReplayBuffer, Sampler
from data import data_utils
from data import shared_storage
from trainer import Trainer

import torch_utils.utils as torch_utils

class Runner(object):
  def __init__(self, config, checkpoint=None, replay_buffer=None):
    self.config = config

    npr.seed(self.config.seed)
    torch.manual_seed(self.config.seed)
    ray.init(num_gpus=self.config.num_gpus, ignore_reinit_error=True)

    # Checkpoint and replay buffer used to share data over processes
    self.checkpoint = {
      'weights' : None,
      'optimizer_state' : None,
      'total_reward' : 0,
      'past_100_rewards' : collections.deque([0] * 100, maxlen=100),
      'eps_len' : 0,
      'mean_value' : 0,
      'eps_obs' : None,
      'eps_values' : None,
      'eps_q_maps' : None,
      'eps_sampled_actions' : None,
      'training_step' : 0,
      'lr' : [0, 0, 0],
      'forward_loss' : 0,
      'state_value_loss' : 0,
      'q_value_loss' : 0,
      'reward_loss' : 0,
      'class_weights': torch.ones(self.config.num_depth_classes),
      'sampled_idxs' : torch.zeros(self.config.max_steps + 1),
      'training_pred': None,
      'eps_reward' : list(),
      'num_eps' : 0,
      'num_steps' : 0,
      'log_counter' : 0,
      'terminate' : False
    }
    self.replay_buffer = dict()

    # Load checkpoint/replay buffer if specified
    if checkpoint:
      checkpoint_path = os.path.join(self.config.root_path,
                                     checkpoint,
                                     'model.checkpoint')
    else:
      checkpoint_path = None

    if replay_buffer:
      replay_buffer_path = os.path.join(self.config.root_path,
                                        replay_buffer,
                                        'replay_buffer.pkl')
    else:
      replay_buffer_path = None

    self.loadCheckpoint(checkpoint_path=checkpoint_path,
                        replay_buffer_path=replay_buffer_path)
    #from data import constants
    #checkpoint = torch.load(constants.RESULTS_PATH + '/block_stacking_3/no_q_actions_32/model.checkpoint')
    #self.checkpoint['weights'] = (checkpoint['weights'][0],
    #                              checkpoint['weights'][1],
    #                              checkpoint['weights'][2])

    if not self.checkpoint['weights']:
      self.initWeights()

    # Workers
    self.data_gen_workers = None
    self.replay_buffer_worker = None
    self.sim_replay_buffer_worker = None
    self.sampler_workers = None
    self.sim_sampler_workers = None
    self.reanalyse_worker = None
    self.shared_storage_worker = None
    self.training_worker = None
    self.test_worker = None

  def initWeights(self):
    device = torch.device('cpu')

    q_value_model = QValueModel(1, 1, device)
    forward_model = ObsPredictionModel(device, self.config.num_depth_classes).to(device)
    state_value_model = StateValueModel(1, device)
    self.checkpoint['weights'] = (torch_utils.dictToCpu(forward_model.state_dict()),
                                  torch_utils.dictToCpu(state_value_model.state_dict()),
                                  torch_utils.dictToCpu(q_value_model.state_dict()))

  def train(self):
    # Init workers
    if self.config.gen_data_on_gpu and self.config.num_gpus == 1:
      trainer_gpu_allocation = 0.4
      sampler_gpu_allocation = 0
      num_data_gen_gpus = (0.6 / self.config.num_agent_workers)
    elif self.config.gen_data_on_gpu and self.config.num_gpus == 4:
      trainer_gpu_allocation = 0.6
      sampler_gpu_allocation = (3 - trainer_gpu_allocation) / self.config.num_sampler_workers
      num_data_gen_gpus = (1.0 / self.config.num_agent_workers)
    else:
      trainer_gpu_allocation = 0.8
      sampler_gpu_allocation = (self.config.num_gpus - trainer_gpu_allocation) / self.config.num_sampler_workers
      num_data_gen_gpus = 0

    self.training_worker = Trainer.options(num_cpus=0, num_gpus=trainer_gpu_allocation).remote(self.checkpoint, self.config)

    self.sampler_workers = [
      Sampler.options(num_cpus=0, num_gpus=sampler_gpu_allocation).remote(self.checkpoint, self.config, self.config.seed + seed)
      for seed in range(self.config.num_sampler_workers)
    ]
    self.replay_buffer_worker = ReplayBuffer.options(num_cpus=0, num_gpus=0, max_concurrency=self.config.num_sampler_workers).remote(self.checkpoint, self.replay_buffer, self.config, self.sampler_workers)

    self.data_gen_workers = [
      DataGenerator.options(num_cpus=0, num_gpus=num_data_gen_gpus).remote(self.checkpoint, self.config, self.config.seed + seed)
      for seed in range(self.config.num_agent_workers)
    ]

    self.expert_data_gen_workers = [
      ExpertDataGenerator.options(num_cpus=0, num_gpus=0).remote(self.checkpoint, self.config, self.config.seed + seed)
      for seed in range(self.config.num_agent_workers)
    ]

    self.shared_storage_worker = shared_storage.SharedStorage.remote(self.checkpoint, self.config)
    self.shared_storage_worker.setInfo.remote('terminate', False)

    # Start workers
    if self.config.num_expert_episodes > 0:
      for data_gen_worker in self.expert_data_gen_workers:
        data_gen_worker.continuousDataGen.remote(self.shared_storage_worker, self.replay_buffer_worker)
    else:
      for data_gen_worker in self.data_gen_workers:
        data_gen_worker.continuousDataGen.remote(self.shared_storage_worker, self.replay_buffer_worker)

    self.training_worker.continuousUpdateWeights.remote(self.replay_buffer_worker, self.shared_storage_worker)

    self.loggingLoop()

  def loggingLoop(self):
    self.save(logging=True)

    self.test_worker = DataGenerator.options(num_cpus=1, num_gpus=0).remote(self.checkpoint, self.config, self.config.seed + self.config.num_agent_workers)
    self.test_worker.continuousDataGen.remote(self.shared_storage_worker, None, True)

    writer = SummaryWriter(self.config.results_path)

    # Log hyperparameters
    hp_table = [
      f'| {key} | {value} |' for key, value in self.config.__dict__.items()
    ]
    writer.add_text('Hyperparameters', '| Parameter | Value |\n|-------|-------|\n' + '\n'.join(hp_table))

    # Log training loop
    counter = self.checkpoint['log_counter']
    keys = [
      'total_reward',
      'past_100_rewards',
      'eps_len',
      'eps_obs',
      'eps_values',
      'eps_sampled_actions',
      'eps_q_maps',
      'mean_value',
      'training_step',
      'lr',
      'q_value_loss',
      'state_value_loss',
      'reward_loss',
      'forward_loss',
      'class_weights',
      'sampled_idxs',
      'training_pred',
      'eps_reward',
      'num_eps',
      'num_steps'
    ]
    info = ray.get(self.shared_storage_worker.getInfo.remote(keys))
    try:
      while info['training_step'] < self.config.training_steps:
        info = ray.get(self.shared_storage_worker.getInfo.remote(keys))

        if info['num_eps'] > self.config.num_expert_episodes and self.expert_data_gen_workers is not None:
          self.switchDataGeneration()

        # Write scalar data to logger
        writer.add_scalar('1.Total_reward/1.Total_reward', info['total_reward'], counter)
        writer.add_scalar('1.Total_reward/2.Mean_value', info['mean_value'], counter)
        writer.add_scalar('1.Total_reward/3.Eps_len', info['eps_len'], counter)
        writer.add_scalar('1.Total_reward/4.Success_rate',
                          np.mean(info['past_100_rewards']) if info['past_100_rewards'] else 0,
                          info['training_step'])
        writer.add_scalar('1.Total_reward/5.Learning_curve',
                          np.mean(info['eps_reward'][-100:]) if info['eps_reward'] else 0,
                          info['num_eps'])
        writer.add_scalar('2.Workers/1.Num_eps', info['num_eps'], counter)
        writer.add_scalar('2.Workers/2.Training_steps', info['training_step'], counter)
        writer.add_scalar('2.Workers/3.Num_steps', info['num_steps'], counter)
        writer.add_scalar('2.Workers/3.Num_steps', info['num_steps'], counter)
        writer.add_scalar('2.Workers/4.Training_steps_per_eps_step_ratio',
                          info['training_step'] / max(1, info['num_steps']), counter)
        writer.add_scalar('2.Workers/5.Forward_learning_rate', info['lr'][0], counter)
        writer.add_scalar('2.Workers/6.State_value_learning_rate', info['lr'][1], counter)
        writer.add_scalar('2.Workers/7.Q_value_learning_rate', info['lr'][2], counter)
        writer.add_scalar('3.Loss/2.State_value_loss', info['state_value_loss'], counter)
        writer.add_scalar('3.Loss/3.Q_Value_loss', info['q_value_loss'], counter)
        writer.add_scalar('3.Loss/4.Reward_loss', info['reward_loss'], counter)
        writer.add_scalar('3.Loss/5.Forward_loss', info['forward_loss'], counter)

        fig, ax = plt.subplots()
        ax.bar(np.arange(self.config.num_depth_classes), info['class_weights'])
        writer.add_figure('3.Loss/6.Class_weights', fig, counter)
        plt.close()

        fig, ax = plt.subplots()
        ax.bar(np.arange(self.config.max_steps + 1), info['sampled_idxs'])
        writer.add_figure('3.Loss/6.Sampled_idxs', fig, counter)
        plt.close()

        # Write episode observations to logger
        eps_obs = info['eps_obs']
        eps_values = info['eps_values']
        eps_q_maps = info['eps_q_maps']
        eps_sampled_actions = info['eps_sampled_actions']
        if eps_obs is not None:
          eps_len = info['eps_len']

          real_obs = np.array([o[2] for o in eps_obs[0]]).reshape(-1, self.config.obs_size, self.config.obs_size)
          pred_obs = np.array([o for o in eps_obs[1]])
          pred_obs = np.vstack([real_obs[0].reshape(1, self.config.obs_size, self.config.obs_size), pred_obs])
          q_maps = np.array(eps_q_maps).reshape(-1, self.config.obs_size, self.config.obs_size)
          obs = np.vstack([real_obs, pred_obs, q_maps])

          fig = plt.figure(figsize=(14, 7))
          gs = ImageGrid(fig, 111, nrows_ncols=(3, eps_len), axes_pad=0.)
          for i, (ax, o) in enumerate(zip(gs, obs)):
            if i  < eps_len * 2:
              ax.imshow(o.squeeze(), cmap='gray')
            else:
              ax.imshow(o.squeeze())
              if eps_sampled_actions[i % eps_len] is not None:
                ax.scatter(eps_sampled_actions[i % eps_len][:,1], eps_sampled_actions[i % eps_len][:,0], c='r', s=1)

            if i < eps_len:
              ax.set_title(eps_values[i])

            ax.tick_params(color='red')
            for spine in ax.spines.values():
              spine.set_edgecolor('red')

            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
          writer.add_figure('1.Total_reward/5.Observations', fig, counter)

        # Write training forward model predictions
        pred_obs = info['training_pred']
        if pred_obs is not None:
          fig, ax = plt.subplots(nrows=1, ncols=2)
          ax[0].imshow(data_utils.convertProbToDepth(pred_obs[0][0], self.config.num_depth_classes).squeeze(), cmap='gray')
          ax[0].title.set_text('{:.3f}'.format(pred_obs[1][0]))
          ax[1].imshow(pred_obs[0][1].squeeze(), cmap='gray')
          ax[1].title.set_text('{:.3f}'.format(pred_obs[1][1]))
          writer.add_figure('3.Loss/7.Forward Predictions', fig, counter)

        counter += 1
        self.shared_storage_worker.setInfo.remote({'log_counter' : counter})
        time.sleep(0.5)
    except KeyboardInterrupt:
      pass

    if self.config.save_model:
      self.shared_storage_worker.saveReplayBuffer.remote(copy.copy(self.replay_buffer))
    self.terminateWorkers()

  def save(self, logging=False):
    if logging:
      print('Checkpointing model at: {}'.format(self.config.results_path))
    self.shared_storage_worker.saveReplayBuffer.remote(copy.copy(self.replay_buffer))
    self.shared_storage_worker.saveCheckpoint.remote()

  def switchDataGeneration(self):
    # Stop expert data generation
    for data_gen_worker in self.expert_data_gen_workers:
      ray.kill(data_gen_worker)
    self.expert_data_gen_workers = None

    # Start on-policy data generation
    for data_gen_worker in self.data_gen_workers:
      data_gen_worker.continuousDataGen.remote(self.shared_storage_worker, self.replay_buffer_worker)

  def terminateWorkers(self):
    if self.shared_storage_worker:
      self.shared_storage_worker.setInfo.remote('terminate', True)
      self.checkpoint = ray.get(self.shared_storage_worker.getCheckpoint.remote())
    if self.replay_buffer_worker:
      self.replay_buffer = ray.get(self.replay_buffer_worker.getBuffer.remote())

    self.expert_data_gen_workers = None
    self.data_gen_workers = None
    self.test_worker = None
    self.training_worker = None
    self.sampler_workers = None
    self.replay_buffer_worker = None
    self.shared_storage_worker = None

  def loadCheckpoint(self, checkpoint_path=None, replay_buffer_path=None):
    if checkpoint_path:
      if os.path.exists(checkpoint_path):
        self.checkpoint = torch.load(checkpoint_path)
        print('Loading checkpoint from {}'.format(checkpoint_path))
      else:
        print('Checkpoint not found at {}'.format(checkpoint_path))

    if replay_buffer_path:
      if os.path.exists(replay_buffer_path):
        with open(replay_buffer_path, 'rb') as f:
          data = pickle.load(f)

        self.replay_buffer = data['buffer']
        self.checkpoint['num_eps'] = data['num_eps']
        self.checkpoint['num_steps'] = data['num_steps']

        print('Loading replay buffer from {}'.format(replay_buffer_path))
      else:
        print('Replay buffer not found at {}'.format(replay_buffer_path))
    else:
        self.checkpoint['num_eps'] = 0
        self.checkpoint['num_steps'] = 0
        self.checkpoint['training_step'] = 0
