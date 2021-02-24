import unittest
from tqdm import tqdm
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from helping_hands_rl_envs import env_factory

class TestBulletHouse4(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 20, 'obs_size': 90, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 6, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'in_hand_mode': 'raw', 'object_scale_range': (0.60, 0.60),
                'hard_reset_freq': 1000, 'physics_mode' : 'fast'}
  planner_config = {'pos_noise': 0, 'rot_noise': 0}

  def testStepLeft(self):
    self.env_config['seed'] = 1
    env = env_factory.createEnvs(1, 'pybullet', 'house_building_4', self.env_config, self.planner_config)
    env.reset()
    for i in range(9, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepsLeft(), i)
    env.close()

  def testPlanner(self):
    self.env_config['render'] = False
    self.env_config['random_orientation'] = True

    env = env_factory.createEnvs(20, 'pybullet', 'house_building_4', self.env_config, self.planner_config)
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