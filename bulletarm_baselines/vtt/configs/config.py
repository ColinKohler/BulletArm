import os
import numpy as np

class Config(object):
  '''
  Base task config.
  '''

  def __init__(self, vision_size=64, encoder='vtt', num_gpus=1):
    # Env
    self.obs_type = ['vision', 'force', 'proprio']
    self.vision_size = vision_size
    self.obs_size = vision_size + 12
    self.vision_channels = 4
    self.force_dim = 6
    self.force_history = 64
    self.max_force = 100
    self.proprio_dim = 5
    self.seq_len = 8

    self.action_sequence = 'pxyzr'
    self.action_dim =  len(self.action_sequence)

    self.workspace = np.array([[0.30, 0.60], [-0.15, 0.15], [-0.01, 0.25]])
    self.view_type = 'camera_side_rgbd'
    self.random_orientation = True
    self.robot = 'panda'
    self.reward_type = 'sparse'

    self.dpos = 1e-3
    self.drot = np.pi / self.dpos

    # Model
    self.z_dim_1 = 32
    self.z_dim_2 = 256
    self.z_dim = self.z_dim_1 + self.z_dim_2
    self.encoder = encoder

    # Training
    self.root_path = 'data'
    self.num_gpus = num_gpus
    self.pre_training_steps = 1000
    self.gen_data_on_gpu = False
    self.per_beta_anneal_steps = None
    self.clip_gradient = False
    self.deterministic = True

    # Occlusions
    self.occlusion_size = 0
    self.num_occlusions = 0

  def getPerBeta(self, step):
    if self.per_beta_anneal_steps:
      anneal_steps = self.per_beta_anneal_steps
    else:
      anneal_steps = self.training_steps

    r = max((anneal_steps - step) / anneal_steps, 0)
    return (self.init_per_beta - self.end_per_beta) * r + self.end_per_beta

  def getEps(self, step):
    if self.eps_anneal_steps:
      anneal_steps = self.eps_anneal_steps
    else:
      anneal_steps = self.training_steps

    r = max((anneal_steps - step) / anneal_steps, 0)
    return (self.init_eps - self.end_eps) * r + self.end_eps
