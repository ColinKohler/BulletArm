import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.asarray([[0, 50],
                        [0, 50],
                        [0, 100]])

env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 50, 'render': True}
envs = env_factory.createEnvs(1, 'numpy', 'block_picking', env_config)

states, obs = envs.reset()
while True:
  plt.imshow(obs.squeeze(), cmap='gray')
  plt.show()

  import ipdb; ipdb.set_trace()
