import unittest
import time
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

from helping_hands_rl_envs import env_factory

class TestBulletDrawerOpening(unittest.TestCase):
  workspace = np.asarray([[0.3, 0.7],
                          [-0.2, 0.2],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyzrrr', 'num_objects': 5, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}

  # env = createHouseBuilding1Env(PyBulletEnv, env_config)()

  def testPlanner(self):
    self.env_config['render'] = True
    num_processes = 1
    env = env_factory.createEnvs(num_processes, 'rl', 'pybullet', 'drawer_block', self.env_config, {})
    total = 0
    s = 0
    step_times = []
    env.reset()
    pbar = tqdm(total=1000)
    while total < 1000:
      t0 = time.time()
      action = env.getNextAction()
      t_plan = time.time() - t0
      states_, in_hands_, obs_, rewards, dones = env.step(action)
      t_action = time.time() - t0 - t_plan
      t = time.time() - t0
      step_times.append(t)

      s += rewards.sum().int().item()

      if dones.sum():
        total += dones.sum().int().item()

        pbar.set_description(
          '{:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
            .format(float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
        )
      pbar.update(dones.sum().int().item())
    env.close()
