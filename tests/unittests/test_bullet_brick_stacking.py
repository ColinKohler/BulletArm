import unittest
import time
import numpy as np
import torch

from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

from helping_hands_rl_envs import env_factory

class TestBulletHouse1(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 3, 'num_cubes': 2, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka'}


  def testPlanner(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 1
    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'brick_stacking', self.env_config)
    env.reset()
    for i in range(3, -1, -1):
      action = env.getNextAction()
      states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepLeft(), i)
    env.close()


  # def testPlanner2(self):
    # self.env_config['render'] = False
    # self.env_config['reward_type'] = 'sparse'
    # self.env_config['random_orientation'] = True
    #
    # env = env_factory.createEnvs(10, 'rl', 'pybullet', 'brick_stacking', self.env_config, {})
    # total = 0
    # s = 0
    # env.reset()
    # while total < 1000:
    #   states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction())
    #   if dones.sum():
    #     s += rewards.sum().int().item()
    #     total += dones.sum().int().item()
    #     print('{}/{}'.format(s, total))