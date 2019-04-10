import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory
from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.block_stacking_env import createBlockStackingEnv

workspace = np.asarray([[0, 250],
                        [0, 250],
                        [0, 500]])

env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 250, 'render': True, 'action_sequence': 'pxy',
              'num_objects': 3, 'seed': np.random.randint(100), 'random_orientation': True}
env = createBlockStackingEnv(NumpyEnv, env_config)()

states, obs = env.reset()
plt.imshow(obs.squeeze())
plt.show()
action = torch.tensor([0, 50, 50])
(states_, obs_), rewards, dones = env.step(action)
plt.imshow(obs_.squeeze()); plt.show()
action = torch.tensor([1, 100, 100])
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
