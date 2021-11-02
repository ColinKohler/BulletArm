import unittest
import time
from tqdm import tqdm
import numpy as np

from helping_hands_rl_envs import env_factory

class TestBulletRampHouse1(unittest.TestCase):
  env_config = {'action_sequence': 'xyzrrrp', 'num_objects': 4}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    num_processes = 1
    env = env_factory.createEnvs(num_processes,  'ramp_house_building_1', self.env_config, self.planner_config)
    while True:
      env.reset()
      print(1)
