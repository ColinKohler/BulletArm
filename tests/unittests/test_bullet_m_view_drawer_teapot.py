import unittest
import time
from tqdm import tqdm
import numpy as np

from helping_hands_rl_envs import env_factory

class TestBulletTwoViewDrawerTeapot(unittest.TestCase):
  workspace = np.asarray([[0.25, 0.75],
                          [-0.25, 0.25],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'xyzrrrp', 'num_objects': 5, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'physics_mode': 'slow', 'in_hand_mode': 'proj', 'hard_reset_freq': 2,
                }

  planner_config = {'random_orientation': True, 'half_rotation': True}

  def testPlanner(self):
    self.env_config['render'] = False

    env = env_factory.createEnvs(20, 'pybullet', 'multi_view_drawer_teapot', self.env_config, self.planner_config)
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
      pbar.update(int(dones.sum()))
    env.close()

  def testReset(self):
    self.env_config['render'] = True
    num_processes = 1
    env = env_factory.createEnvs(num_processes, 'rl', 'pybullet', 'tilt_block_stacking', self.env_config, {})
    while True:
      states, hand_obs, depths = env.reset()
      print(1)
