import unittest
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from helping_hands_rl_envs.envs.house_building_3_env import createHouseBuilding3Env
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs import env_factory

workspace = np.asarray([[0.35, 0.65],
                          [-0.15, 0.15],
                          [0, 0.50]])
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 90, 'render': True, 'fast_mode': True,
              'seed': 0, 'action_sequence': 'pxyr', 'num_objects': 4, 'random_orientation': True,
              'reward_type': 'sparse', 'simulate_grasp': True, 'perfect_grasp': False, 'robot': 'ur5_robotiq',
              'workspace_check': 'point', 'in_hand_mode': 'raw'}

env = env_factory.createEnvs(1, 'rl', 'pybullet', 'house_building_3', env_config, {})
total = 0
s = 0
env.reset()
while total < 1000:
  states_, in_hands_, obs_, rewards, dones = env.step(env.getNextAction())
  # plt.imshow(in_hands_.squeeze())
  # plt.show()
  if dones.sum():
    s += rewards.sum().int().item()
    total += dones.sum().int().item()
    print('{}/{}'.format(s, total))