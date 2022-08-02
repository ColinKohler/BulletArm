import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from bulletarm import env_factory

def run(task, robot):
  if 'close_loop' in task or 'force' in task:
    env_config = {'robot' : robot, 'render' : True, 'action_sequence' : 'pxyzr', 'view_type': 'camera_center_xyz', 'physics_mode' : 'force'}
    planner_config = {'dpos': 0.05, 'drot': np.pi/4}
  else:
    env_config = {'robot' : robot, 'render' : True}
    planner_config = None
  env = env_factory.createEnvs(0, task, env_config, planner_config)

  obs = env.reset()
  done = False
  while not done:
    action = env.getNextAction()
    obs, reward, done = env.step(action)
    s, in_hand, obs, force = obs

    force1 = uniform_filter1d(np.clip(force, -20, 20) / 20, size=256, axis=0)[-256:]

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(force1[:,0], label='Fx')
    ax.plot(force1[:,1], label='Fy')
    ax.plot(force1[:,2], label='Fz')
    ax.plot(force1[:,3], label='Mx')
    ax.plot(force1[:,4], label='My')
    ax.plot(force1[:,5], label='Mz')

    plt.legend()
    plt.show()

  #force1 = uniform_filter1d(force, size=256, axis=0)
  #force2 = uniform_filter1d(np.clip(force, -20, 20) / 20, size=256, axis=0)

  #fig, ax = plt.subplots(nrows=1, ncols=2)
  #ax[0].plot(force1[:,0], label='Fx')
  #ax[0].plot(force1[:,1], label='Fy')
  #ax[0].plot(force1[:,2], label='Fz')
  #ax[0].plot(force1[:,3], label='Mx')
  #ax[0].plot(force1[:,4], label='My')
  #ax[0].plot(force1[:,5], label='Mz')

  #ax[1].plot(force2[:,0], label='Fx')
  #ax[1].plot(force2[:,1], label='Fy')
  #ax[1].plot(force2[:,2], label='Fz')
  #ax[1].plot(force2[:,3], label='Mx')
  #ax[1].plot(force2[:,4], label='My')
  #ax[1].plot(force2[:,5], label='My')

  #plt.legend()
  #plt.show()

  env.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to run')
  parser.add_argument('--robot', type=str, default='kuka',
    help='Robot to run')

  args = parser.parse_args()
  run(args.task, args.robot)
