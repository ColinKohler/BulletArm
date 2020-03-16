import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
import torch

from helping_hands_rl_envs import env_factory

class TestBulletHouse4(unittest.TestCase):
  workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
  env_config = {'workspace': workspace, 'max_steps': 20, 'obs_size': 90, 'render': True, 'fast_mode': True,
                'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 4, 'random_orientation': True,
                'reward_type': 'step_left', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'kuka',
                'workspace_check': 'point', 'in_hand_mode': 'raw'}


  def testPlanner(self):
    self.env_config['seed'] = 0
    num_random_o = 2
    self.env_config['num_random_objects'] = num_random_o
    self.env_config['render'] = True
    env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_4', self.env_config)
    env.reset()
    for i in range(9, -1, -1):
      action = env.getNextAction()
      states_, in_hands_, obs_, rewards, dones = env.step(action, auto_reset=False)
      if i == 0:
        self.assertTrue(dones)
      else:
        self.assertFalse(dones)
      self.assertEqual(env.getStepLeft(), i)
    env.close()

  def testPlanner2(self):
    self.env_config['render'] = False
    self.env_config['reward_type'] = 'sparse'
    self.env_config['random_orientation'] = True
    self.env_config['robot'] = 'kuka'
    env = env_factory.createEnvs(10, 'rl', 'pybullet', 'house_building_4', self.env_config, {})
    total = 0
    s = 0
    env.reset()
    while total < 1000:
      t0 = time.time()
      action = env.getNextAction()
      # print('plan time: {}'.format(time.time()-t0))
      t0 = time.time()
      states_, in_hands_, obs_, rewards, dones = env.step(action)
      # print('step time: {}'.format(time.time()-t0))
      # plt.imshow(in_hands_.squeeze())
      # plt.colorbar()
      # plt.show()
      if dones.sum():
        s += rewards.sum().int().item()
        total += dones.sum().int().item()
        print('{}/{}'.format(s, total))

    ## 0.837 kuka
    ## 0.951 ur5
    ## 0.950 ur5 robotiq