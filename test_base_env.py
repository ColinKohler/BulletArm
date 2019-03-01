import time
import pybullet as pb
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import torch

import env_factory


def test_vrep():
  vrep_workspace = np.asarray([[-0.25, 0.25],
                               [0.25, 0.75],
                               [0, 0.4]])
  env_config = {'workspace': vrep_workspace, 'max_steps': 10, 'obs_size': 250, 'fast_mode': True, 'render': False,
                'port': 21000, 'action_sequence': 'xyzr'}
  envs = env_factory.createEnvs(1, 'vrep', 'block_picking', env_config)

  states, obs = envs.reset()

  for i in range(0, 8):
    plt.imshow(np.squeeze(obs[0]), cmap='gray')
    plt.show()

    # actions = torch.tensor([[0, 0.0, 0.35, i*(2*np.pi/8)]])
    actions = torch.tensor([[0.0, 0.35, 0.1, i*(2*np.pi/8)]])
    states_, obs_, rewards, dones = envs.step(actions)

def test_pybullet():
  pybullet_workspace = np.asarray([[0.25, 0.75],
                                   [-0.25, 0.25],
                                   [0, 0.4]])

  env_config = {'workspace': pybullet_workspace, 'max_steps': 10, 'obs_size': 250, 'fast_mode': False, 'render': True,
                'action_sequence': 'xyz'}

  envs = env_factory.createEnvs(1, 'pybullet', 'block_picking', env_config)

  states, obs = envs.reset()

  for i in range(0, 8):
    plt.imshow(np.squeeze(obs[0]), cmap='gray')
    plt.show()

    # actions = torch.tensor([[0, 0.0, 0.35, i*(2*np.pi/8)]])
    actions = torch.tensor([[0.35, 0.0, 0.1]])
    states_, obs_, rewards, dones = envs.step(actions)

if __name__ == '__main__':
  # test_pybullet()
  test_vrep()