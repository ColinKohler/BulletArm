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
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 8, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka'}

  # env = createHouseBuilding1Env(PyBulletEnv, env_config)()

  def testStepLeft(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'improvise_house_building_2', self.env_config, {})
    env.reset()

    positions = env.getObjPositions()[0]
    # pick up the roof
    action = [0, positions[1][0], positions[1][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)


    env.close()
