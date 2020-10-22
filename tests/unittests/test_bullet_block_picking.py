import unittest
import time
import numpy as np

import matplotlib.pyplot as plt

from helping_hands_rl_envs import env_factory

class TestBulletBlockStacking(unittest.TestCase):
  workspace = np.asarray([[0.3, 0.6],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzr', 'num_objects': 4, 'random_orientation': True,
                'reward_type': 'dense', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'pybullet', 'block_picking', self.env_config, self.planner_config)
    env.reset()
    for i in range(3, -1, -1):
      action = env.getNextAction()
      states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepsLeft(), i)
      self.assertEqual(rewards, 1)
    env.close()

