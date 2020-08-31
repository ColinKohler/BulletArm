import unittest
import time
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

from helping_hands_rl_envs import env_factory

class TestBulletDrawerOpening(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 5, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}

  # env = createHouseBuilding1Env(PyBulletEnv, env_config)()

  def testPlanner(self):
    self.env_config['render'] = True
    num_processes = 1
    env = env_factory.createEnvs(num_processes, 'rl', 'pybullet', 'drawer_opening', self.env_config, {})
    while True:
      states, hand_obs, depths = env.reset()
      action = torch.tensor([[0, 0.52, 0, 0.06, 0, -np.pi/6, 0]])
      states_, in_hands_, obs_, rewards, dones = env.step(action)
      plt.imshow(obs_.squeeze())
      plt.show()
      print(1)
