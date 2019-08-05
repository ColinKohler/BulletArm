import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory
from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.block_cylinder_stacking_env import createBlockCylinderStackingEnv

workspace = np.asarray([[0, 100],
                        [0, 100],
                        [0, 500]])

env_config = {'workspace': workspace, 'max_steps': 4, 'obs_size': 100, 'render': True, 'action_sequence': 'pxyrz',
              'num_objects': 4, 'seed': np.random.randint(100), 'random_orientation': False, 'reward_type': 'sparse'}
env = createBlockCylinderStackingEnv(NumpyEnv, env_config)()
states, obs = env.reset()
plt.imshow(obs.squeeze())
plt.show()

pos = env.getObjectPosition()

action = np.array([0, pos[0][0], pos[0][1], 0, 0])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze())
plt.show()
action = np.array([1, pos[1][0], pos[1][1], 0, 8])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze())
plt.show()
action = np.array([0, pos[2][0], pos[2][1], 0, 0])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze())
plt.show()
action = np.array([1, pos[3][0], pos[3][1], 0, 8])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze())
plt.show()
pass
