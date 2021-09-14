import sys
sys.path.append('..')

import os
import datetime
import math
import numpy as np

from data import constants
from data.configs.config import Config

class BottleTrayConfig(Config):
  def __init__(self, num_gpus=1, results_path=None):
    super(BottleTrayConfig, self).__init__(num_gpus=num_gpus)
    self.seed = 0

    # Env
    self.env_type = 'bottle_tray'
    self.expert_env = 'bottle_tray'
    self.use_rot = True
    self.num_depth_classes = 11
    self.max_height= 0.2
    self.workspace = constants.WORKSPACE
    self.obs_resolution = constants.OBS_RESOLUTION
    self.obs_size = constants.BOTTLE_TRAY_ENV_CONFIG['obs_size']
    self.hand_obs_size = constants.BOTTLE_TRAY_ENV_CONFIG['in_hand_size']
    self.max_steps = constants.BOTTLE_TRAY_ENV_CONFIG['max_steps']

    # Agent
    self.depth = 1
    self.num_sampled_actions = 10
    self.num_rots = 8

    # Data gen
    self.num_agent_workers = 4
    self.num_expert_episodes = 100
    self.discount = 0.95

    # Exploration strategy
    self.init_action_sample_pen_size = [0, 2]
    self.end_action_sample_pen_size = [0, 2]
    self.decay_action_sample_pen = 2000

    # Training
    self.root_path = os.path.join(constants.RESULTS_PATH, 'bottle_tray')
    self.expert_data_path = None
    if results_path:
      self.results_path = os.path.join(self.root_path, results_path)
    else:
      self.results_path = os.path.join(self.root_path, datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S"))
    self.save_model = True
    self.training_steps = 10000
    self.batch_size = 32
    self.checkpoint_interval = 100

    # LR schedule
    self.forward_lr_init = 1e-3
    self.forward_weight_decay = 1e-6
    self.state_value_lr_init = 1e-3
    self.state_value_weight_decay = 1e-5
    self.q_value_lr_init = 1e-4
    self.q_value_weight_decay = 1e-5
    self.lr_decay = 0.95
    self.decay_lr_interval = 2000

    # Replay buffer
    self.replay_buffer_size = 5000
    self.num_unroll_steps = 1
    self.per_alpha = 0.6
    self.init_per_beta = 0.0
    self.end_per_beta = 1.0
    self.per_eps = 1e-6
    self.expert_per_eps = 1

    # Wait times
    self.data_gen_delay = 0
    self.training_delay = 0
    self.train_data_ratio = 0
