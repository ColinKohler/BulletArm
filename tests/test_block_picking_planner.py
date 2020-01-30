import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.array([[0.35, 0.65], [-0.15, 0.15], [0, 1]])
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'action_sequence': 'pxy',
              'num_objects': 1, 'render': True, 'fast_mode': True}
envs = env_factory.createEnvs(1, 'data', 'pybullet', 'block_picking', env_config)

state, obs = envs.reset()
done = False
while not done:
  action = envs.getNextAction()
  print(action)
  state_, obs_, reward, done = envs.step(action)

  obs = obs_
