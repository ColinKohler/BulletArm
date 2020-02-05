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
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka'}


  def testPlanner(self):
    # env = createHouseBuilding3Env(PyBulletEnv, self.env_config)()
    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config)
    env.reset()
    for i in range(5, -1, -1):
      action = env.getNextAction()
      states_, in_hands_, obs_, rewards, dones = env.step(action)
      self.assertEqual(rewards.item(), i)
    env.close()


  def testPlanner2(self):
    self.env_config['render'] = False
    self.env_config['reward_type'] = 'sparse'
    self.env_config['random_orientation'] = True
    env = env_factory.createEnvs(10, 'rl', 'pybullet', 'house_building_3', self.env_config, {})
    total = 0
    s = 0
    env.reset()
    while total < 1000:
      states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction())
      if dones.sum():
        s += rewards.sum().int().item()
        total += dones.sum().int().item()
        print('{}/{}'.format(s, total))

  def testBlockNotValidTriangleOnBrick(self):
    self.env_config['render'] = False
    self.env_config['random_orientation'] = False

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config, {})
    env.reset()

    env.save()
    position = env.getObjPositions()[0]
    action = torch.tensor([0, position[2][0], position[2][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 7)

    action = torch.tensor([1, position[3][0], position[3][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 8)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 7)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 6)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 5)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 4)
    env.restore()
    env.close()

  def testBlockNotValidBrickOrRoofOnBlock(self):
    self.env_config['seed'] = 1
    self.env_config['render'] = True
    self.env_config['random_orientation'] = False

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config)
    env.reset()

    env.save()
    position = list(env.getObjPositions())[0]
    action = torch.tensor([0, position[3][0], position[3][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 7)

    action = torch.tensor([1, position[0][0], position[0][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 8)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 7)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 6)
    env.restore()

    position = list(env.getObjPositions())[0]
    action = torch.tensor([0, position[2][0], position[2][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 7)

    action = torch.tensor([1, position[0][0], position[0][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 8)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 7)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 6)

    env.close()

  def testBlockNotValidRoofOnBrickOnBlock(self):
    self.env_config['random_orientation'] = False

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config)
    env.reset()

    env.save()
    position = list(env.getObjPositions())[0]
    action = torch.tensor([0, position[3][0], position[3][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 7)

    action = torch.tensor([1, position[0][0], position[0][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 8)

    position = list(env.getObjPositions())[0]
    action = torch.tensor([0, position[2][0], position[2][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 9)

    action = torch.tensor([1, position[0][0], position[0][1], 0]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 10)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 9)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 8)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 7)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 6)

    env.close()

  def testBlockValidTriangleOnBrick(self):
    self.env_config['random_orientation'] = False

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config)
    env.reset()

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 5)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 4)

    position = list(env.getObjPositions())[0]
    action = torch.tensor([0, position[2][0], position[2][1], np.pi / 2]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 5)

    action = torch.tensor([1, position[3][0], position[3][1], np.pi / 2]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 6)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 5)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 4)

    env.close()

  def testBlockValidBrickOrRoofOnBlock(self):
    self.env_config['random_orientation'] = False

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config)
    env.reset()

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 5)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 4)

    env.save()
    position = list(env.getObjPositions())[0]
    action = torch.tensor([0, position[3][0], position[3][1], np.pi / 2]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 3)

    action = torch.tensor([1, position[0][0], position[0][1], np.pi / 2]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 4)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 3)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 2)
    env.restore()

    position = list(env.getObjPositions())[0]
    action = torch.tensor([0, position[2][0], position[2][1], np.pi / 2]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 5)

    action = torch.tensor([1, position[0][0], position[0][1], np.pi / 2]).unsqueeze(0)
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 6)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 5)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 4)

    env.close()

  def testSuccess(self):
    self.env_config['random_orientation'] = False

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', self.env_config)
    env.reset()

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 5)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 4)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 3)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 2)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 1)

    action = env.getNextAction()
    states_, in_hands_, obs_, rewards, dones = env.step(action)
    self.assertEqual(rewards, 0)

    env.close()
