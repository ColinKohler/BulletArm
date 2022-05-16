import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletPyramidStacking(unittest.TestCase):
  env_config = {}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 0
    env = env_factory.createEnvs(1,  'pyramid_stacking', self.env_config, self.planner_config)
    env.reset()
    for i in range(3, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepsLeft(), i)
    env.close()

  def testPlanner2(self):
    self.env_config['render'] = False
    self.env_config['seed'] = 0
    self.env_config['physics_mode'] = 'fast'
    num_processes = 20
    env = env_factory.createEnvs(num_processes,  'pyramid_stacking', self.env_config, self.planner_config)
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

      pbar.set_description(
        '{}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
    env.close()