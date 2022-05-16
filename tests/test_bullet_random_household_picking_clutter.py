import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletBlockStacking(unittest.TestCase):
  workspace_size = 0.4
  workspace = np.asarray([[0.5-workspace_size/2, 0.5+workspace_size/2],
                          [0-workspace_size/2, 0+workspace_size/2],
                          [0, 0+workspace_size]])
  env_config = {'workspace': workspace, 'max_steps': 30, 'obs_size': 128, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 15, 'random_orientation': True,
                'reward_type': 'dense', 'simulate_grasp': True, 'perfect_grasp': True, 'robot': 'kuka',
                'workspace_check': 'point', 'object_scale_range': (0.5, 0.5),
                'min_object_distance': 0., 'min_boarder_padding': 0.15, 'adjust_gripper_after_lift': True
                }

  planner_config = {'random_orientation': True}

  def testPlanner2(self):
    self.env_config['render'] = True
    self.env_config['random_orientation'] = True
    self.env_config['seed'] = 0
    num_processes = 1
    env = env_factory.createEnvs(num_processes,  'random_household_picking_clutter', self.env_config, self.planner_config)
    total = 0
    s = 0
    steps = 0
    step_times = []
    obs = env.reset()
    pbar = tqdm(total=1000)
    while total < 1000:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=True)
      s += rewards.sum()
      print(rewards)
      total += dones.sum()
      steps += num_processes
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)
      if self.env_config['reward_type'] == 'dense':
        sr = float(s) / steps if total != 0 else 0
      else:
        sr = float(s) / total if total != 0 else 0

      pbar.set_description(
        '{:.2f}/{}, avg: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, steps, sr, t_plan, t_action, np.mean(step_times))
      )
    env.close()
