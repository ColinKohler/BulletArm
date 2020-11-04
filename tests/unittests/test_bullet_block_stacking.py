import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from helping_hands_rl_envs import env_factory

class TestBulletBlockStacking(unittest.TestCase):
  workspace = np.asarray([[0.3, 0.6],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 4, 'random_orientation': True,
                'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'pybullet', 'block_stacking', self.env_config, self.planner_config)
    env.reset()
    for i in range(5, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepsLeft(), i)
    self.assertEqual(rewards, 1)
    self.assertEqual(dones, 1)
    env.close()

  def testPlanner2(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 0
    self.env_config['physics_mode'] = 'fast'
    num_processes = 5
    env = env_factory.createEnvs(num_processes, 'pybullet', 'block_stacking', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    env.reset()
    pbar = tqdm(total=1000)
    while total < 1000:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      (states_, in_hands_, obs_), rewards, dones = env.step(action)
      s += rewards.sum()
      total += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
    env.close()

