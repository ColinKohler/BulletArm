import unittest
import time
import numpy as np

import matplotlib.pyplot as plt

from helping_hands_rl_envs import env_factory

class TestBulletHouse4Deconstruct(unittest.TestCase):
  env_config = {}
  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = False
    env = env_factory.createEnvs(1,  'house_building_4_deconstruct', self.env_config, self.planner_config)
    env.reset()
    for i in range(9, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(dones, 1)
    env.close()

