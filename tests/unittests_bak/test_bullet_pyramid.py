import unittest
import time
import numpy as np
import torch

from helping_hands_rl_envs.envs.pyramid_stacking_env import createPryamidStackingEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs import env_factory

class TestBulletPyramid(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 120, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxy', 'num_objects': 3, 'random_orientation': False,
                'simulate_grasp': True, 'perfect_grasp': False, 'workspace_check': 'point'}


  def testPlanner(self):
    # env = createHouseBuilding3Env(PyBulletEnv, self.env_config)()
    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'pyramid_stacking', self.env_config)
    env.reset()
    for i in range(5, -1, -1):
      action = env.getNextAction()
      states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepLeft(), i)
    env.close()

  def testSuccess(self):
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

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepLeft(), 2)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepLeft(), 1)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepLeft(), 0)

    env.close()
