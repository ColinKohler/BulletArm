import unittest
import time
from tqdm import tqdm
import numpy as np
import torch

from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

from helping_hands_rl_envs import env_factory

class TestBulletHouse2(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 5, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}

  # env = createHouseBuilding1Env(PyBulletEnv, env_config)()

  def testPlanner(self):
    # self.env_config['render'] = True
    #
    # env = env_factory.createEnvs(1, 'rl', 'pybullet', 'improvise_house_building_4', self.env_config, {})
    # total = 0
    # s = 0
    # step_times = []
    # env.reset()
    # pbar = tqdm(total=1000)
    # while total < 1000:
    #   t0 = time.time()
    #   action = env.getNextAction()
    #   t_plan = time.time() - t0
    #   states_, in_hands_, obs_, rewards, dones = env.step(action)
    #   t_action = time.time() - t0 - t_plan
    #   t = time.time() - t0
    #   step_times.append(t)
    #
    #   if dones.sum():
    #     s += rewards.sum().int().item()
    #     total += dones.sum().int().item()
    #
    #   pbar.set_description(
    #     '{}/{}, SR: {:.3f}, plan time: {:.2f}, action time: {:.2f}, avg step time: {:.2f}'
    #       .format(s, total, float(s) / total if total != 0 else 0, t_plan, t_action, np.mean(step_times))
    #   )
    #   pbar.update(dones.sum().int().item())
    # env.close()

    self.env_config['render'] = True

    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'improvise_house_building_4', self.env_config, {})
    while True:
      env.reset()
