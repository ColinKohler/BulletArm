import unittest
import time
import numpy as np
import torch

from helping_hands_rl_envs import env_factory

class TestBulletHouse1(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 3, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}
  planner_config = {'pos_noise': 0, 'rot_noise': 0}

  def testStepLeft(self):
    num_random_o = 0
    self.env_config['num_random_objects'] = num_random_o
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'pybullet', 'house_building_1', self.env_config, self.planner_config)
    env.reset()

    position = env.getObjectPositions()[0]
    action = [[0, position[0+num_random_o][0], position[0+num_random_o][1], 0]]
    states_, in_hands_, obs_, rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)
    self.assertEqual(dones, 0)

    states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction())
    self.assertEqual(env.getStepsLeft(), 4)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[0, position[1+num_random_o][0], position[1+num_random_o][1], 0]]
    states_, in_hands_, obs_, rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 3)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[1, position[1+num_random_o][0], position[1+num_random_o][1], 0]]
    states_, in_hands_, obs_, rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 2)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[0, position[1+num_random_o][0], position[1+num_random_o][1], 0]]
    states_, in_hands_, obs_, rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 3)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[1, position[1+num_random_o][0], position[1+num_random_o][1], 0]]
    states_, in_hands_, obs_, rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 2)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[0, position[0+num_random_o][0], position[0+num_random_o][1], 0]]
    states_, in_hands_, obs_, rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 1)
    self.assertEqual(dones, 0)

    states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 0)
    self.assertEqual(dones, 1)
    env.close()
