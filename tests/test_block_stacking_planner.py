import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.array([[0.35, 0.65], [-0.15, 0.15], [0, 1]])
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'action_sequence': 'pxy',
              'num_objects': 3, 'render': False, 'fast_mode': True}
planner_config = {'pos_noise': 0.020}
envs = env_factory.createEnvs(1, 'data', 'pybullet', 'block_stacking', env_config, planner_config=planner_config)

state, hand_obs, obs = envs.reset()
done = False
while not done:
  plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()
  action = envs.getNextAction()
  state_, hand_obs_, obs_, reward, done, valid = envs.step(action)

  print(envs.didBlockFall())

  obs = obs_
  hand_obs = hand_obs_

plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()
