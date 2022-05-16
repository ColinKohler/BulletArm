import unittest
from tqdm import tqdm
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletHouse4(unittest.TestCase):
  env_config = {}
  planner_config = {'pos_noise': 0, 'rot_noise': 0}

  def testStepLeft(self):
    self.env_config['seed'] = 1
    env = env_factory.createEnvs(1,  'house_building_4', self.env_config, self.planner_config)
    env.reset()
    for i in range(9, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepsLeft(), i)
    env.close()

  def testPlanner(self):
    self.env_config['render'] = False
    self.env_config['random_orientation'] = True

    env = env_factory.createEnvs(20,  'house_building_4', self.env_config, self.planner_config)
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