import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from bulletarm import env_factory

def run(task, robot):
  if 'close_loop' in task or 'force' in task:
    env_config = {'robot' : robot, 'render' : True, 'action_sequence' : 'pxyzr', 'view_type': 'render_center', 'physics_mode' : 'force', 'max_steps' : 50, 'obs_size' : 128}
    planner_config = {'dpos': 0.025, 'drot': np.pi/8}
  else:
    env_config = {'robot' : robot, 'render' : True}
    planner_config = None
  env = env_factory.createEnvs(0, task, env_config, planner_config)

  s = 0
  for _ in range(20):
    s, in_hand, obs, force = env.reset()
    done = False
    action_his_len = [force.shape[0]]
    while not done:
      action = env.getNextAction()
      obs, reward, done = env.step(action)
      s, in_hand, obs, force = obs
      action_his_len.append(force.shape[0])

      fig, ax = plt.subplots(nrows=1, ncols=2)
      ax[0].imshow(obs.squeeze(), cmap='gray')
      ax[1].plot(force[:,0], label='Fx')
      ax[1].plot(force[:,1], label='Fy')
      ax[1].plot(force[:,2], label='Fz')
      ax[1].plot(force[:,3], label='Mx')
      ax[1].plot(force[:,4], label='My')
      ax[1].plot(force[:,5], label='Mz')
      ax[1].set_ylim(-1,1)
      plt.legend()
      plt.show()

    print(reward)
    s += reward

  print(s)
  max_force = 30
  smooth_force = np.clip(force, -max_force, max_force) / max_force
  smooth_force_1 = np.clip(uniform_filter1d(force, size=64, axis=0), -max_force, max_force) / max_force
  smooth_force_2 = np.tanh(uniform_filter1d(force * 0.1, size=64, axis=0))

  print(np.diff(action_his_len))

  fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

  ax[0].title.set_text('Clip [-{}, {}]'.format(max_force, max_force))
  ax[0].plot(smooth_force[:,0], label='Fx')
  ax[0].plot(smooth_force[:,1], label='Fy')
  ax[0].plot(smooth_force[:,2], label='Fz')
  ax[0].plot(smooth_force[:,3], label='Mx')
  ax[0].plot(smooth_force[:,4], label='My')
  ax[0].plot(smooth_force[:,5], label='Mz')

  ax[1].title.set_text('Smoothing N=64, Clip [-{}, {}]'.format(max_force, max_force))
  ax[1].plot(smooth_force_1[:,0], label='Fx')
  ax[1].plot(smooth_force_1[:,1], label='Fy')
  ax[1].plot(smooth_force_1[:,2], label='Fz')
  ax[1].plot(smooth_force_1[:,3], label='Mx')
  ax[1].plot(smooth_force_1[:,4], label='My')
  ax[1].plot(smooth_force_1[:,5], label='Mz')

  ax[2].title.set_text('Smoothing N=64, tanh'.format(max_force, max_force))
  ax[2].plot(smooth_force_2[:,0], label='Fx')
  ax[2].plot(smooth_force_2[:,1], label='Fy')
  ax[2].plot(smooth_force_2[:,2], label='Fz')
  ax[2].plot(smooth_force_2[:,3], label='Mx')
  ax[2].plot(smooth_force_2[:,4], label='My')
  ax[2].plot(smooth_force_2[:,5], label='Mz')

  for i in range(3):
    for a in action_his_len:
      ax[i].axvline(x=a, color='black', linestyle='--')

  plt.legend()
  plt.subplot_tool()
  plt.show()

  env.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to run')
  parser.add_argument('--robot', type=str, default='kuka',
    help='Robot to run')

  args = parser.parse_args()
  run(args.task, args.robot)
