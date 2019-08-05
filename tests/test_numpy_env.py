import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.asarray([[0, 100],
                        [0, 100],
                        [0, 500]])

env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 100, 'render': True, 'action_sequence': 'pxy',
              'num_objects': 3}
envs = env_factory.createEnvs(1, 'numpy', 'block_stacking', env_config)

states, obs = envs.reset()
for i in range(1, 3):
  plt.imshow(obs.squeeze(), cmap='gray')
  plt.show()
  actions = torch.tensor([[i * 10, i * 10]])
  states_, obs_, rewards, dones = envs.step(actions)

  obs = obs_
  pass
