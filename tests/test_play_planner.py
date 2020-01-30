import time
import numpy as np
import matplotlib.pyplot as plt
import torch

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.array([[0.35, 0.65], [-0.15, 0.15], [0, 1]])
env_config = {'workspace': workspace, 'max_steps': 5, 'obs_size': 128, 'action_sequence': 'pxy',
              'num_objects': 5, 'planner': 'play', 'render': False, 'fast_mode': True}
envs = env_factory.createEnvs(1, 'data', 'pybullet', 'block_stacking', env_config)

state, in_hand_img, obs = envs.reset()
done = False
obses = [obs]; in_hand_imgs = [in_hand_img]
while not done:

  action = envs.getNextAction()
  state_, in_hand_img_, obs_, reward, done = envs.step(action)

  obs = obs_
  in_hand_img = in_hand_img_

  obses.append(obs_)
  in_hand_imgs.append(in_hand_img_)

fig, ax = plt.subplots(nrows=len(obses), ncols=2, figsize=(10,15))
for i in range(len(obses)):
  ax[i][0].imshow(obses[i].squeeze(), cmap='gray', vmin=0.0, vmax=0.1)
  ax[i][1].imshow(in_hand_imgs[i].squeeze(), cmap='gray', vmin=0.0, vmax=0.1)
plt.show()
