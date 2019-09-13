import unittest
import time
import numpy as np

from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

class TestBulletHouse1(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 1000, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 3, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka'}

  env = createHouseBuilding1Env(PyBulletEnv, env_config)()

  def testStepLeft(self):
    env = self.env
    states, obs = env.reset()

    position = list(env.getObjectPosition())
    action = [0, position[0][0], position[0][1], np.pi / 2]
    (states_, obs_), rewards, dones = env.step(np.array(action))
    self.assertEqual(rewards, 5)
    self.assertEqual(dones, 0)

    action = env.getPlan()
    (states_, obs_), rewards, dones = env.step(np.array(action))
    self.assertEqual(rewards, 4)
    self.assertEqual(dones, 0)

    position = list(env.getObjectPosition())
    action = [0, position[1][0], position[1][1], 0]
    (states_, obs_), rewards, dones = env.step(np.array(action))
    self.assertEqual(rewards, 3)
    self.assertEqual(dones, 0)

    position = list(env.getObjectPosition())
    action = [1, position[2][0], position[2][1], 0]
    (states_, obs_), rewards, dones = env.step(np.array(action))
    self.assertEqual(rewards, 2)
    self.assertEqual(dones, 0)

    position = list(env.getObjectPosition())
    action = [0, position[1][0], position[1][1], 0]
    (states_, obs_), rewards, dones = env.step(np.array(action))
    self.assertEqual(rewards, 3)
    self.assertEqual(dones, 0)

    position = list(env.getObjectPosition())
    action = [1, position[2][0], position[2][1], 0]
    (states_, obs_), rewards, dones = env.step(np.array(action))
    self.assertEqual(rewards, 2)
    self.assertEqual(dones, 0)

    position = list(env.getObjectPosition())
    action = [0, position[0][0], position[0][1], 0]
    (states_, obs_), rewards, dones = env.step(np.array(action))
    self.assertEqual(rewards, 1)
    self.assertEqual(dones, 0)

    action = env.getPlan()
    (states_, obs_), rewards, dones = env.step(np.array(action))
    self.assertEqual(rewards, 0)
    self.assertEqual(dones, 1)
