import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.array([[0, 128], [0, 128], [0, 100]])
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'action_sequence': 'pxy',
              'num_objects': 2, 'render': False}
envs = env_factory.createEnvs(1, 'data', 'numpy', 'block_stacking', env_config)

state, obs = envs.reset()
done = False
while not done:
  plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=15); plt.show()
  action = envs.getNextAction()
  state_, obs_, reward, done = envs.step(action)

  obs = obs_
plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=15); plt.show()
