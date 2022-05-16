import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletCloseLoopBlockPicking(unittest.TestCase):
  env_config = {}

  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi / 4}

  def testPlanner2(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 1
    num_processes = 1
    env = env_factory.createEnvs(num_processes,  'close_loop_block_picking', self.env_config, self.planner_config)
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
      # plt.imshow(obs_[0, 0], vmin=0, vmax=0.2)
      # plt.show()
      s += rewards.sum()
      total += dones.sum()
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      pbar.set_description(
        '{:.3f}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
    env.close()

