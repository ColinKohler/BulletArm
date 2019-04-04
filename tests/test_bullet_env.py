import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from helping_hands_rl_envs.envs.block_picking_env import createBlockPickingEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

workspace = np.asarray([[0.25, 0.75],
                        [-0.25, 0.25],
                        [0, 0.50]])
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 50, 'render': True, 'fast_mode': False, 'seed': 1, 'action_sequence': 'pxyzr'}
env = createBlockPickingEnv(PyBulletEnv, env_config)()

total = 0
s = 0
while True:
  states, obs = env.reset()
  position = env.getObjectPosition()[0]
  # env.step((0, 0.505, 0.066, 0))
  obs_, rewards, dones = env.step((0, position[0], position[1], 0.02, 0))
  s += int(rewards)
  total += 1
  print('{}/{}'.format(s, total))
  # env.step((1, 0.475, 0.066, 0))
  # import ipdb; ipdb.set_trace()
