import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from bulletarm import env_factory

class TestBulletImproviseHouse3Deconstruct(unittest.TestCase):
  env_config = {}
  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 0
    env = env_factory.createEnvs(1,  'improvise_house_building_3_deconstruct', self.env_config, self.planner_config)
    env.reset()
    for i in range(5, -1, -1):
      action = env.getNextAction()
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
    self.assertEqual(dones, 1)
    env.close()

  def testPlanner2(self):
    self.env_config['render'] = True
    num_processes = 1
    env = env_factory.createEnvs(num_processes,  'improvise_house_building_3_deconstruct', self.env_config, self.planner_config)
    total = 0
    s = 0
    step_times = []
    env.reset()
    pbar = tqdm(total=1000)
    steps = [0 for i in range(num_processes)]
    while total < 1000:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=False)
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      steps = list(map(lambda x: x+1, steps))
      num_objects = [len(p) for p in env.getObjectPositions()]

      for i in range(num_processes):
        if dones[i]:
          if steps[i] == 2*(num_objects[i]-1):
            s += 1
          total += 1
          steps[i] = 0
      done_idxes = np.nonzero(dones)[0]
      if done_idxes.shape[0] != 0:
        env.reset_envs(done_idxes)

      pbar.set_description(
        '{}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
          .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
      )
      pbar.update(total-pbar.n)
    env.close()