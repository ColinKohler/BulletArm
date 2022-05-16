import unittest
import time
import numpy as np

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletBlockPicking(unittest.TestCase):
  env_config = {'num_objects': 4}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'block_picking', self.env_config, self.planner_config)
    env.reset()
    for i in range(3, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepsLeft(), i)
    self.assertEqual(rewards, 1)
    env.close()

