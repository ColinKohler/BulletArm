import unittest
import time
from tqdm import tqdm
import numpy as np

from helping_hands_rl_envs import env_factory

class TestBulletRampHouse1Deconstruct(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'xyzrrrp', 'num_objects': 4, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'in_hand_mode': 'proj', 'object_scale_range': (0.6, 0.6)}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    num_processes = 1
    env = env_factory.createEnvs(num_processes, 'pybullet', 'ramp_house_building_1', self.env_config, self.planner_config)
    while True:
      env.reset()
      print(1)
