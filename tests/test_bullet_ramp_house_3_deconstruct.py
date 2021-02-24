import unittest
import time
from tqdm import tqdm
import numpy as np
import torch

from helping_hands_rl_envs import env_factory

class TestBulletRampHouseBuilding3Deconstruct(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'xyzrrrp', 'num_objects': 4, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'in_hand_mode': 'proj', 'object_scale_range': (0.6, 0.6)}

  planner_config = {'random_orientation': True}

  def testPlanner(self):
    self.env_config['render'] = False
    num_processes = 20
    env = env_factory.createEnvs(num_processes, 'pybullet', 'ramp_house_building_3_deconstruct', self.env_config, self.planner_config)
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
