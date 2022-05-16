import unittest
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletHouse3(unittest.TestCase):
  env_config = {}
  planner_config = {'pos_noise': 0, 'rot_noise': 0}

  def testPlanner(self):
    # env = createHouseBuilding3Env(PyBulletEnv, self.env_config)()
    env = env_factory.createEnvs(1,  'house_building_3', self.env_config, self.planner_config)
    env.reset()
    for i in range(5, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepsLeft(), i)
    env.close()


  # def testPlanner2(self):
  #   self.env_config['render'] = False
  #   self.env_config['reward_type'] = 'sparse'
  #   self.env_config['random_orientation'] = True
  #   self.env_config['robot'] = 'ur5_robotiq'
  #   env = env_factory.createEnvs(10, 'rl',  'house_building_3', self.env_config, {})
  #   total = 0
  #   s = 0
  #   env.reset()
  #   while total < 1000:
  #     (states_, in_hands_, obs_), rewards, dones = env.step(env.getNextAction())
  #     # plt.imshow(in_hands_.squeeze())
  #     # plt.show()
  #     if dones.sum():
  #       s += rewards.sum().int().item()
  #       total += dones.sum().int().item()
  #       print('{}/{}'.format(s, total))

    ## 0.992 kuka
    ## 0.989 ur5
    ## 0.995 ur5 robotiq

  def testBlockNotValidTriangleOnBrick(self):
    self.env_config['seed'] = 0
    self.env_config['random_orientation'] = False

    env = env_factory.createEnvs(1,  'house_building_3', self.env_config, self.planner_config)
    env.reset()

    env.save()
    position = env.getObjectPositions()[0]
    action = np.array([[0, position[0][0], position[0][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 7)

    action = np.array([[1, position[1][0], position[1][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 8)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 7)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 6)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 4)
    env.restore()
    env.close()

  def testBlockNotValidBrickOrRoofOnBlock(self):
    self.env_config['seed'] = 0
    self.env_config['random_orientation'] = False

    env = env_factory.createEnvs(1,  'house_building_3', self.env_config, self.planner_config)
    env.reset()

    env.save()
    position = list(env.getObjectPositions())[0]
    action = np.array([[0, position[0][0], position[0][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 7)

    action = np.array([[1, position[2][0], position[2][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 8)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 7)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 6)
    env.restore()

    position = list(env.getObjectPositions())[0]
    action = np.array([[0, position[1][0], position[1][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 7)

    action = np.array([[1, position[2][0], position[2][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 8)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 7)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 6)

    env.close()

  def testBlockNotValidRoofOnBrickOnBlock(self):
    self.env_config['seed'] = 1
    self.env_config['random_orientation'] = False
    self.env_config['render'] = True

    env = env_factory.createEnvs(1,  'house_building_3', self.env_config, self.planner_config)
    env.reset()

    env.save()
    position = list(env.getObjectPositions())[0]
    action = np.array([[0, position[1][0], position[1][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 7)

    action = np.array([[1, position[2][0], position[2][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 8)

    position = list(env.getObjectPositions())[0]
    action = np.array([[0, position[0][0], position[0][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 9)

    action = np.array([[1, position[2][0], position[2][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 10)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 9)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 8)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 7)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 6)

    env.close()

  def testBlockValidTriangleOnBrick(self):
    self.env_config['seed'] = 1
    self.env_config['random_orientation'] = False
    self.env_config['render'] = True

    env = env_factory.createEnvs(1,  'house_building_3', self.env_config, self.planner_config)
    env.reset()

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 4)

    position = list(env.getObjectPositions())[0]
    action = np.array([[0, position[0][0], position[0][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)

    action = np.array([[1, position[1][0], position[1][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 6)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 4)

    env.close()

  def testBlockValidBrickOrRoofOnBlock(self):
    self.env_config['seed'] = 1
    self.env_config['random_orientation'] = False
    self.planner_config['random_orientation'] = False
    self.env_config['render'] = True

    env = env_factory.createEnvs(1,  'house_building_3', self.env_config, self.planner_config)
    env.reset()

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 4)

    env.save()
    position = list(env.getObjectPositions())[0]
    action = np.array([[0, position[1][0], position[1][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 3)

    action = np.array([[1, position[2][0], position[2][1], np.pi/2]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 4)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 3)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 2)
    env.restore()

    position = list(env.getObjectPositions())[0]
    action = np.array([[0, position[0][0], position[0][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)

    action = np.array([[1, position[2][0], position[2][1], 0]])
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 6)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 4)

    env.close()

  def testSuccess(self):
    self.env_config['seed'] = 0
    self.env_config['random_orientation'] = False
    self.env_config['render'] = True

    env = env_factory.createEnvs(1,  'house_building_3', self.env_config, self.planner_config)
    env.reset()

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 4)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 3)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 2)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 1)

    action = env.getNextAction()
    (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 0)

    env.close()
