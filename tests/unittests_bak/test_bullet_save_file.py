import unittest
import time
import numpy as np
import torch

from helping_hands_rl_envs.envs.house_building_3_env import createHouseBuilding3Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs import env_factory

class TestBulletHouse3(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 4, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}

  def test(self):
    self.env_config['render'] = False
    self.env_config['seed'] = 2
    self.env_config['random_orientation'] = False

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config)
    env.reset()

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepLeft(), 5)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepLeft(), 4)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepLeft(), 3)

    env.saveToFile('save')
    env.close()
    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config)
    env.reset()
    env.loadFromFile('save')

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepLeft(), 2)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepLeft(), 1)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepLeft(), 0)

    env.saveToFile('save')
    env.close()
    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config)
    env.reset()
    env.loadFromFile('save')

    env.close()
