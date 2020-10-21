import time
import numpy as np
import matplotlib.pyplot as plt

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.array([[0.35, 0.65], [-0.15, 0.15], [0, 1]])
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': 128, 'action_sequence': 'pxy',
              'num_objects': 3, 'render': False, 'fast_mode': True}
planner_config = {'pos_noise': 0.005, 'rand_pick_prob': 0., 'rand_place_prob': 0., 'gamma': 0.75}
envs = env_factory.createEnvs(1, 'data', 'pybullet', 'pyramid_stacking', env_config, planner_config=planner_config)

for i in range(100):
  state, hand_obs, obs = envs.reset()
  done = False
  while not done:
    plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()
    action = envs.getNextAction()
    state_, hand_obs_, obs_, reward, done, valid = envs.step(action)

    obs = obs_
    hand_obs = hand_obs_
    print(envs.getStepsLeft())

  plt.imshow(obs.squeeze(), cmap='gray', vmin=0.0, vmax=0.1); plt.show()
