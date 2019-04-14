import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory
from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.block_stacking_env import createBlockStackingEnv
from helping_hands_rl_envs.envs.block_picking_env import createBlockPickingEnv

workspace = np.asarray([[0, 100],
                        [0, 100],
                        [0, 500]])

env_config = {'workspace': workspace, 'max_steps': 1, 'obs_size': 100, 'render': True, 'action_sequence': 'pxyrz',
              'num_objects': 1, 'seed': np.random.randint(100), 'random_orientation': True, 'reward_type': 'sparse'}
env = createBlockPickingEnv(NumpyEnv, env_config)()
states, obs = env.reset()
plt.imshow(obs.squeeze())
plt.show()

pos = env.getObjectPosition()
rewards = np.zeros((8, 8))
for i in range(8):
  for j in range(8):
    env.saveState()
    action = torch.tensor([0, pos[0][0], pos[0][1], i*(np.pi/8), j*(10./8)])
    (states_, obs_), reward, dones = env.step(action)
    rewards[i, j] = reward
    # plt.imshow(obs_.squeeze());
    # plt.show()
    env.restoreState()

plt.imshow(obs.squeeze())
plt.show()
action = torch.tensor([0, pos[0][0], pos[0][1]])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze()); plt.show()

env.saveState()

action = torch.tensor([0, pos[1][0], pos[1][1]])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze()); plt.show()

env.restoreState()

action = torch.tensor([0, 100, 100])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze()); plt.show()

action = torch.tensor([0, 150, 150])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze()); plt.show()

env.saveState()

action = torch.tensor([1, 100, 100])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze()); plt.show()

env.restoreState()

action = torch.tensor([1, 100, 100])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze()); plt.show()

action = torch.tensor([0, 100, 100])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze()); plt.show()

action = torch.tensor([1, 50, 50])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze()); plt.show()

obs = obs_
pass
