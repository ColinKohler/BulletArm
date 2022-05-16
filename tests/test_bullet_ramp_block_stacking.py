import unittest
import time
from tqdm import tqdm
import numpy as np

from bulletarm import env_factory

class TestBulletRampBlockStacking(unittest.TestCase):
  env_config = {'action_sequence': 'xyzrrrp', 'num_objects': 4}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = True

    env = env_factory.createEnvs(1,  'ramp_block_stacking', self.env_config, self.planner_config)
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

  def testReset(self):
    self.env_config['render'] = True
    num_processes = 1
    env = env_factory.createEnvs(num_processes,  'ramp_block_stacking', self.env_config, self.planner_config)
    while True:
      states, hand_obs, depths = env.reset()
      print(1)
