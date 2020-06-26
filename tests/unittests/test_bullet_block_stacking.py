import unittest
import time
import numpy as np
import torch

from helping_hands_rl_envs.envs.house_building_1_env import createHouseBuilding1Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
import matplotlib.pyplot as plt

from helping_hands_rl_envs import env_factory

class TestBulletHouse1(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': False, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 4, 'random_orientation': False,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point'}
  heightmap_resolution = 0.3 / 90

  def testPlanner(self):
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'block_stacking', self.env_config)
    env.reset()
    for i in range(5, -1, -1):
      action = env.getNextAction()
      states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
      self.assertEqual(env.getStepLeft(), i)
    env.close()


  def testPlanner2(self):
    self.env_config['render'] = False
    self.env_config['reward_type'] = 'sparse'
    self.env_config['random_orientation'] = True
    self.env_config['num_objects'] = 4

    env = env_factory.createEnvs(20, 'rl', 'pybullet', 'block_stacking', self.env_config, {'half_rotation': True})
    total = 0
    s = 0
    states, in_hands, obs = env.reset()
    while total < 1000:
      action = env.getNextAction()
      states_, in_hands_, obs_, rewards, dones = env.step(action)

      # pixel_x = ((action[0, 1] - self.workspace[0][0]) / self.heightmap_resolution).long()
      # pixel_y = ((action[0, 2] - self.workspace[1][0]) / self.heightmap_resolution).long()
      # pixel_x = torch.clamp(pixel_x, 0, 90 - 1).item()
      # pixel_y = torch.clamp(pixel_y, 0, 90 - 1).item()
      # fig, axs = plt.subplots(1, 2, figsize=(10,5))
      # axs[0].imshow(obs.squeeze())
      # axs[1].imshow(obs_.squeeze())
      # axs[0].scatter(pixel_y, pixel_x, c='r')
      # axs[1].scatter(pixel_y, pixel_x, c='r')
      # fig.show()

      obs = obs_
      if dones.sum():
        s += rewards.sum().int().item()
        total += dones.sum().int().item()
        print('{}/{}'.format(s, total))