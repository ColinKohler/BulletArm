import unittest
import time
import numpy as np
import torch

from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

from helping_hands_rl_envs import env_factory

class TestBulletHouse2(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 5, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka'}

  # env = createHouseBuilding1Env(PyBulletEnv, env_config)()

  def testPlanner(self):
    self.env_config['render'] = True

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'improvise_house_building_2', self.env_config, {})
    total = 0
    s = 0
    env.reset()
    while total < 1000:
      states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction())
      if dones.sum():
        s += rewards.sum().int().item()
        total += dones.sum().int().item()
        print('{}/{}'.format(s, total))
    env.close()
