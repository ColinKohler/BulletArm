import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.asarray([[0, 250],
                        [0, 250],
                        [0, 500]])

env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 250, 'render': True, 'action_sequence': 'xy',
              'num_objects': 3}
envs = env_factory.createEnvs(1, 'numpy', 'block_picking', env_config)

states, obs = envs.reset()
for i in range(1, 5):
  plt.imshow(obs.squeeze(), cmap='gray')
  plt.show()
  actions = torch.tensor([[i * 50, i * 50]])
  states_, obs_, rewards, dones = envs.step(actions)
  obs = obs_
  pass

