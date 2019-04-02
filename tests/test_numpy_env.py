import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.asarray([[0, 250],
                        [0, 250],
                        [0, 500]])

env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 250, 'render': True, 'action_sequence': 'xyr'}
envs = env_factory.createEnvs(1, 'numpy', 'block_picking', env_config)

for i in range(8):
  states, obs = envs.reset()
  plt.imshow(obs.squeeze(), cmap='gray')
  actions = torch.tensor([[100, 200, i * np.pi/8]])
  plt.show()
  states_, obs_, rewards, dones = envs.step(actions)
  pass

