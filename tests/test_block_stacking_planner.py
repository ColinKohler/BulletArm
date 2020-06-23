import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.array([[0.35, 0.65], [-0.15, 0.15], [0, 1]])
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 256, 'action_sequence': 'pxy',
              'num_objects': 3, 'render': False, 'fast_mode': True, 'simulate_grasps': True, 'robot': 'kuka'}
planner_config = {'pos_noise': 0.030, 'rand_pick_prob': 0.0, 'rand_place_prob': 0.0}
envs = env_factory.createEnvs(1, 'data', 'pybullet', 'block_stacking', env_config, planner_config=planner_config)

success = 0
falls = 0
other = 0
for _ in range(100):
  state, hand_obs, obs = envs.reset()
  done = False
  s = 0
  while not done:
    s+=1
    plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()
    action = envs.getNextAction()
    state_, hand_obs_, obs_, reward, done, valid = envs.step(action)

    if done and reward:
      success += 1
    if done and not reward and not envs.didBlockFall().item():
      other += 1
    if envs.didBlockFall().item():
      falls += 1
      done = True

    obs = obs_
    hand_obs = hand_obs_

  plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()

print(success)
print(other)
print(falls)
