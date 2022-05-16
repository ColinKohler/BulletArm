import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletCloseLoopClutterPicking(unittest.TestCase):
  env_config = {}

  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi / 4}

  def testPlanner2(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 5
    num_processes = 1
    env = env_factory.createEnvs(num_processes, 'close_loop_clutter_picking', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    (states, in_hands, obs) = env.reset()
    pbar = tqdm(total=100)
    while total < 100:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
      # axs[0].imshow(obs[0, 0])
      # axs[1].imshow(obs[0, 1])
      # fig.show()

      plt.imshow(obs[0, 0], vmin=0, vmax=0.25)
      plt.colorbar()
      plt.show()
      t0 = time.time()
      (states_, in_hands_, obs_), rewards, dones = env.simulate(action)
      plt.imshow(obs_[0, 0], vmin=0, vmax=0.25)
      plt.show()

      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      plt.imshow(obs_[0, 0], vmin=0, vmax=0.25)
      plt.show()
      if rewards:
        print(1)
      obs = obs_
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

