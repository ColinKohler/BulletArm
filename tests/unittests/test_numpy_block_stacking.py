import unittest
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

from helping_hands_rl_envs import env_factory

class TestBulletHouse1(unittest.TestCase):
  workspace = np.asarray([[0, 90],
                          [0, 90],
                          [0, 90]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 4, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka'}


  def testPlanner(self):
    env = env_factory.createEnvs(1, 'rl', 'numpy', 'block_stacking', self.env_config)
    states_, in_hands_, obs_ = env.reset()
    plt.imshow(obs_.squeeze())
    plt.show()
    for i in range(5, -1, -1):
      action = env.getNextAction()
      states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
      plt.imshow(obs_.squeeze())
      plt.show()
      self.assertEqual(env.getStepLeft(), i)
    env.close()


  # def testPlanner2(self):
  #   self.env_config['render'] = False
  #   self.env_config['reward_type'] = 'sparse'
  #   self.env_config['random_orientation'] = True
  #   self.env_config['num_objects'] = 4
  #
  #   env = env_factory.createEnvs(10, 'rl', 'pybullet', 'house_building_1', self.env_config, {})
  #   total = 0
  #   s = 0
  #   env.reset()
  #   while total < 1000:
  #     states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction())
  #     if dones.sum():
  #       s += rewards.sum().int().item()
  #       total += dones.sum().int().item()
  #       print('{}/{}'.format(s, total))