import unittest
import time
from tqdm import tqdm
import numpy as np
import torch

from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

from helping_hands_rl_envs import env_factory

class TestBulletH5(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 6, 'random_orientation': False,
                'reward_type': 'dense', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}

  # env = createHouseBuilding1Env(PyBulletEnv, env_config)()

  def test(self):
    self.env_config['render'] = True

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_5', self.env_config, {})
    env.reset()
    position = env.getObjPositions()[0]
    action = [0, position[0][0], position[0][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(dones, 0)
    self.assertEqual(rewards, 0)

    position = env.getObjPositions()[0]
    action = [1, position[2][0], position[2][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(dones, 0)
    self.assertEqual(rewards, 1)

    position = env.getObjPositions()[0]
    action = [0, position[0][0], position[0][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(dones, 0)
    self.assertEqual(rewards, -1)

    position = env.getObjPositions()[0]
    action = [1, position[2][0], position[2][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(dones, 0)
    self.assertEqual(rewards, 1)

    position = env.getObjPositions()[0]
    action = [0, position[1][0], position[1][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(dones, 0)
    self.assertEqual(rewards, 0)

    position = env.getObjPositions()[0]
    action = [1, position[3][0], position[3][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(dones, 0)
    self.assertEqual(rewards, 1)

    position = env.getObjPositions()[0]
    action = [0, position[2][0], position[2][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(dones, 0)
    self.assertEqual(rewards, 0)

    position = env.getObjPositions()[0]
    action = [1, position[4][0], position[4][1], 0]
    states_, in_hands_, obs_, rewards, dones = env.step(torch.tensor(action).unsqueeze(0), auto_reset=False)
    self.assertEqual(dones, 1)
    self.assertEqual(rewards, 1)