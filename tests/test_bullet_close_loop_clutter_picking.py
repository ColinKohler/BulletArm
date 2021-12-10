import unittest
import time
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from helping_hands_rl_envs import env_factory

class TestBulletBowlStacking(unittest.TestCase):
  workspace = np.asarray([[0.3, 0.6],
                          [-0.15, 0.15],
                          [0.01, 0.25]])
  env_config = {'workspace': workspace, 'max_steps': 20, 'obs_size': 128, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzr', 'num_objects': 5, 'random_orientation': True,
                'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'panda',
                'workspace_check': 'point', 'physics_mode': 'fast', 'hard_reset_freq': 1000, 'object_scale_range': (0.8, 1),
                'view_type': 'camera_center_xyz', 'transparent_bin': False, 'collision_penalty': False}

  planner_config = {'random_orientation': True, 'dpos': 0.05, 'drot': np.pi/8}

  def testPlanner2(self):
    self.env_config['render'] = True
    self.env_config['seed'] = 0
    num_processes = 1
    env = env_factory.createEnvs(num_processes, 'pybullet', 'close_loop_clutter_picking', self.env_config, self.planner_config)
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

      # plt.imshow(obs[0, 0], vmin=-0.1, vmax=0.35)
      # plt.colorbar()
      # plt.show()
      t0 = time.time()
      (states_, in_hands_, obs_), rewards, dones = env.simulate(action)

      (states_, in_hands_, obs_), rewards, dones = env.step(action, auto_reset=True)
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

