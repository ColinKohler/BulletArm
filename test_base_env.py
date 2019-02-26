import time
import pybullet as pb
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import torch

import env_factory

workspace = np.asarray([[-0.25, 0.25],
                        [0.25, 0.75],
                        [0, 0.4]])

# env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 250, 'fast_mode': False, 'render': True}
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 250, 'fast_mode': True, 'render': True}
envs = env_factory.createEnvs(1, 'vrep', 'block_picking', env_config)

states, obs = envs.reset()

while True:
  plt.imshow(np.squeeze(obs[0]), cmap='gray'); plt.show()
  y = np.random.uniform(0.25, 0.75)
  x = np.random.uniform(-0.25, 0.25)
  actions = torch.tensor([[0, x, y, np.pi/4]])
  states_, obs_, rewards, dones = envs.step(actions)
  import ipdb; ipdb.set_trace()
