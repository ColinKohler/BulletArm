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
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 3, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka'}

  # env = createHouseBuilding1Env(PyBulletEnv, env_config)()

  def testStepLeft(self):
    num_random_o = 2
    self.env_config['num_random_objects'] = num_random_o
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_2', self.env_config, {})
    env.reset()

    positions = env.getObjPositions()[0]
    # pick up the roof
    action = [0, positions[2+num_random_o][0], positions[2+num_random_o][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(env.getStepLeft(), 5)
    self.assertEqual(dones, 0)

    states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepLeft(), 4)
    self.assertEqual(dones, 0)

    positions = env.getObjPositions()[0]
    action = [0, positions[1+num_random_o][0], positions[1+num_random_o][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(env.getStepLeft(), 3)
    self.assertEqual(dones, 0)

    states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepLeft(), 2)
    self.assertEqual(dones, 0)

    states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepLeft(), 1)
    self.assertEqual(dones, 0)

    states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepLeft(), 0)
    self.assertEqual(dones, 1)

    env.close()


  # def testPlanner2(self):
  #   self.env_config['render'] = False
  #   self.env_config['reward_type'] = 'sparse'
  #   self.env_config['random_orientation'] = True
  #   env = env_factory.createEnvs(10, 'rl', 'pybullet', 'house_building_2', self.env_config, {})
  #   total = 0
  #   s = 0
  #   env.reset()
  #   while total < 1000:
  #     states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction())
  #     if dones.sum():
  #       s += rewards.sum().int().item()
  #       total += dones.sum().int().item()
  #       print('{}/{}'.format(s, total))