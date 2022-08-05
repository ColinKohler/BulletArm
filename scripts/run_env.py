import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from bulletarm import env_factory

def run(task, robot):
  if 'close_loop' in task or 'force' in task:
    env_config = {'robot' : robot, 'render' : True, 'action_sequence' : 'pxyzr', 'view_type': 'camera_center_xyz', 'physics_mode' : 'force', 'max_steps' : 100}
    planner_config = {'dpos': 0.05, 'drot': np.pi/4}
  else:
    env_config = {'robot' : robot, 'render' : True}
    planner_config = None
  env = env_factory.createEnvs(0, task, env_config, planner_config)

  s, in_hand, obs, force = env.reset()
  done = False
  action_his_len = [force.shape[0]]
  while not done:
    action = env.getNextAction()
    obs, reward, done = env.step(action)
    s, in_hand, obs, force = obs
    #action_his_len.append(force.shape[0])

    #plt.plot(force[:,0], label='Fx')
    #plt.plot(force[:,1], label='Fy')
    #plt.plot(force[:,2], label='Fz')
    #plt.plot(force[:,3], label='Mx')
    #plt.plot(force[:,4], label='My')
    #plt.plot(force[:,5], label='Mz')
    #plt.ylim(-1,1)
    #plt.legend()
    #plt.show()

    #plt.imshow(obs.squeeze(), cmap='gray'); plt.show()

  #max_force = 10
  #smooth_force = np.clip(force, -max_force, max_force) / max_force
  #smooth_force_1 = np.clip(uniform_filter1d(force, size=32, axis=0), -max_force, max_force) / max_force
  #smooth_force_2 = np.tanh(uniform_filter1d(force, size=32, axis=0))

  #print(np.diff(action_his_len))

  #fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

  #ax[0].title.set_text('Clip [-{}, {}]'.format(max_force, max_force))
  #ax[0].plot(smooth_force[:,0], label='Fx')
  #ax[0].plot(smooth_force[:,1], label='Fy')
  #ax[0].plot(smooth_force[:,2], label='Fz')
  #ax[0].plot(smooth_force[:,3], label='Mx')
  #ax[0].plot(smooth_force[:,4], label='My')
  #ax[0].plot(smooth_force[:,5], label='Mz')

  #ax[1].title.set_text('Smoothing N=32, Clip [-{}, {}]'.format(max_force, max_force))
  #ax[1].plot(smooth_force_1[:,0], label='Fx')
  #ax[1].plot(smooth_force_1[:,1], label='Fy')
  #ax[1].plot(smooth_force_1[:,2], label='Fz')
  #ax[1].plot(smooth_force_1[:,3], label='Mx')
  #ax[1].plot(smooth_force_1[:,4], label='My')
  #ax[1].plot(smooth_force_1[:,5], label='Mz')

  #ax[2].title.set_text('Smoothing N=64, Clip [-{}, {}]'.format(max_force, max_force))
  #ax[2].plot(smooth_force_2[:,0], label='Fx')
  #ax[2].plot(smooth_force_2[:,1], label='Fy')
  #ax[2].plot(smooth_force_2[:,2], label='Fz')
  #ax[2].plot(smooth_force_2[:,3], label='Mx')
  #ax[2].plot(smooth_force_2[:,4], label='My')
  #ax[2].plot(smooth_force_2[:,5], label='Mz')

  #for i in range(3):
  #  for a in action_his_len:
  #    ax[i].axvline(x=a, color='black', linestyle='--')

  #plt.legend()
  #plt.subplot_tool()
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
