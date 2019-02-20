import time
import pybullet as pb
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import torch

import env_factory

workspace = np.asarray([[0.25, 0.75],
                        [-0.25, 0.25],
                        [0, 0.4]])

env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 250, 'fast_mode': False}
envs = env_factory.createEnvs(1, 'pybullet', 'block_picking', env_config)

states, obs = envs.reset()

while True:
  plt.imshow(np.squeeze(obs[0]), cmap='gray'); plt.show()
  actions = torch.tensor([[0, 0.35, 0.0, 0.0]])
  states_, obs_, rewards, dones = envs.step(actions)
  import ipdb; ipdb.set_trace()
