import time
import numpy as np
import matplotlib.pyplot as plt

import helping_hands_rl_envs.env_factory as env_factory

workspace = np.array([[0.30, 0.60], [-0.15, 0.15], [0, 1]])
obs_size = 300
plot = True
render = False
env_config = {'workspace': workspace, 'max_steps': 10, 'obs_size': obs_size, 'action_sequence': 'pxy',
              'num_objects': 3, 'render': render, 'fast_mode': True, 'simulate_grasps': True, 'robot': 'kuka', 'seed':10}
planner_config = {'pick_noise': [0.0, 0.0], 'place_noise': [0.0, 0.0], 'rand_pick_prob': 0.0, 'rand_place_prob': 0.0, 'planner_res': 3}
envs = env_factory.createEnvs(1, 'data', 'pybullet', 'block_stacking', env_config, planner_config=planner_config)
heightmap_resolution = (workspace[0,1] - workspace[0,0]) / obs_size

success = 0
falls = 0
other = 0
import time
t0 = time.time()
for sim in range(100):
  state, hand_obs, obs = envs.reset()
  done = False
  s = 0
  while not done:
    s+=1
    action = envs.getNextAction()
    state_, hand_obs_, obs_, reward, done, valid = envs.step(action)
    x_pix = (action[0,2].item() - workspace[1][0]) / heightmap_resolution
    y_pix = (action[0,1].item() - workspace[0][0]) / heightmap_resolution

    if plot:
      fig, ax = plt.subplots(nrows=1, ncols=4)
      ax[0].imshow(obs.squeeze(), cmap='gray', vmin=0.0)
      ax[0].scatter([x_pix], [y_pix], s=1, c='r')
      ax[1].imshow(obs_.squeeze(), cmap='gray', vmin=0.0)
      ax[1].scatter([x_pix], [y_pix], s=1, c='r')
      ax[2].imshow(hand_obs.squeeze(), cmap='gray', vmin=0.0)
      ax[3].imshow(hand_obs_.squeeze(), cmap='gray', vmin=0.0)
      plt.show()

    if done and reward and not envs.didBlockFall().item():
      print('{}: s'.format(sim))
      success += 1
    if done and not reward and not envs.didBlockFall().item():
      print('{}: f'.format(sim))
      other += 1
    if envs.didBlockFall().item():
      print('{}: f'.format(sim))
      falls += 1
      done = True

    obs = obs_
    hand_obs = hand_obs_

print(time.time()-t0)
print(success)
print(other)
print(falls)
