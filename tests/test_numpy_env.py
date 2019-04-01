import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.asarray([[0, 50],
                        [0, 50],
                        [0, 50]])

env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 50, 'render': True, 'action_sequence': 'xyzr'}
envs = env_factory.createEnvs(1, 'numpy', 'block_picking', env_config)

for i in range(100):
  states, obs = envs.reset()
  plt.imshow(obs.squeeze(), cmap='gray')
  actions = torch.tensor([[100, 200, i * 10, i * np.pi/8]])
  plt.show()
  states_, obs_, rewards, dones = envs.step(actions)
  pass

