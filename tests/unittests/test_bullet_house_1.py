import unittest
import time
import numpy as np
import torch
from tqdm import tqdm

from helping_hands_rl_envs import env_factory

class TestBulletHouse1(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 3, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}
  planner_config = {'pos_noise': 0, 'rot_noise': 0, 'random_orientation': False}

  def testStepLeft(self):
    num_random_o = 0
    self.env_config['num_random_objects'] = num_random_o
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'pybullet', 'house_building_1', self.env_config, self.planner_config)
    env.reset()

    position = env.getObjectPositions()[0]
    action = [[0, position[0+num_random_o][0], position[0+num_random_o][1], 0]]
    (states_, in_hands_, obs_), rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 5)
    self.assertEqual(dones, 0)

    (states_, in_hands_, obs_), rewards, dones = env.step(env.getNextAction())
    self.assertEqual(env.getStepsLeft(), 4)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[0, position[1+num_random_o][0], position[1+num_random_o][1], 0]]
    (states_, in_hands_, obs_), rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 3)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[1, position[1+num_random_o][0], position[1+num_random_o][1], 0]]
    (states_, in_hands_, obs_), rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 2)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[0, position[1+num_random_o][0], position[1+num_random_o][1], 0]]
    (states_, in_hands_, obs_), rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 3)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[1, position[1+num_random_o][0], position[1+num_random_o][1], 0]]
    (states_, in_hands_, obs_), rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 2)
    self.assertEqual(dones, 0)

    position = env.getObjectPositions()[0]
    action = [[0, position[0+num_random_o][0], position[0+num_random_o][1], 0]]
    (states_, in_hands_, obs_), rewards, dones = env.step(np.array(action), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 1)
    self.assertEqual(dones, 0)

    (states_, in_hands_, obs_), rewards, dones = env.step(env.getNextAction(), auto_reset=False)
    self.assertEqual(env.getStepsLeft(), 0)
    self.assertEqual(dones, 1)
    env.close()

  def testPlanner(self):
    self.env_config['render'] = True
    self.env_config['random_orientation'] = True
    self.env_config['num_objects'] = 5

    env = env_factory.createEnvs(1, 'pybullet', 'house_building_1', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    env.reset()
    pbar = tqdm(total=1000)
    while total < 1000:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=True)
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      s += rewards.sum()

      if dones.sum():
        total += dones.sum()

        pbar.set_description(
          '{:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
            .format(float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
        )
      pbar.update(dones.sum())
    env.close()
