import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from helping_hands_rl_envs import env_factory

class TestBulletBumpyHouseBuilding4(unittest.TestCase):
  env_config = {'action_sequence': 'pxyzrrr'}

  planner_config = {'random_orientation': True, 'half_rotation': True}


  def testPlanner2(self):
    self.env_config['render'] = True
    num_processes = 1
    self.env_config['seed'] = 0
    env = env_factory.createEnvs(num_processes,  'bumpy_house_building_4', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    env.reset()
    pbar = tqdm(total=500)
    while total < 500:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0

      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=True)

      # (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      # if dones:
      #   env.reset()

      s += rewards.sum()
      total += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
      pbar.update(dones.sum())
    env.close()

