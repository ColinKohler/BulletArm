import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from helping_hands_rl_envs import env_factory

class TestBulletCovidTest(unittest.TestCase):
  # workspace = np.asarray([[0.1, 0.7],
  #                         [-0.3, 0.3],
  #                         [0, 0.50]])

  workspace_size = 0.4
  workspace = np.asarray([[0.5 - workspace_size / 2, 0.5 + workspace_size / 2],
                          [0 - workspace_size / 2, 0 + workspace_size / 2],
                          [0, 0 + workspace_size]])  #????????????????????
  env_config = {'workspace': workspace, 'max_steps': 27, 'obs_size': 128, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 3, 'random_orientation': True,
                'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'fast', 'hard_reset_freq': 1000,
                'object_scale_range': (0.6, 0.6), 'pick_top_down_approach': True, 'place_top_down_approach': True}

  planner_config = {'random_orientation': True, 'half_rotation': False}

  def testPlanner2(self):
    self.env_config['render'] = False
    self.env_config['seed'] = 0
    num_processes = 10
    env = env_factory.createEnvs(num_processes, 'pybullet', 'covid_test', self.env_config, self.planner_config)
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
      s += rewards.sum()
      total += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.update(dones.sum())
      pbar.set_description(
        '{}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
    env.close()

