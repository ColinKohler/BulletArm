import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from bulletarm import env_factory

def run(task, robot):
  workspace = np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]])
  if 'close_loop' in task or 'force' in task:
    env_config = {'robot' : robot, 'render' : True, 'action_sequence' : 'pxyzr', 'workspace' : workspace, 'view_type': 'render_center', 'physics_mode' : 'force', 'max_steps' : 50, 'obs_size' : 32}
    planner_config = {'dpos': 0.05, 'drot': np.pi/4}
  else:
    env_config = {'robot' : robot, 'render' : True}
    planner_config = None
  env = env_factory.createEnvs(0, task, env_config, planner_config)

  for _ in range(20):
    s, in_hand, obs, force = env.reset()
    done = False
    action_his_len = [force.shape[0]]
    while not done:
      action = env.getNextAction()
      obs, reward, done = env.step(action)
      s, in_hand, obs, force = obs
      action_his_len.append(force.shape[0])

      print(np.mean(np.abs(force)))

      fig, ax = plt.subplots(nrows=1, ncols=2)
      ax[0].imshow(obs.squeeze(), cmap='gray')
      ax[1].plot(force[:,0], label='wFx')
      ax[1].plot(force[:,1], label='wFy')
      ax[1].plot(force[:,2], label='wFz')
      ax[1].plot(force[:,3], label='wMx')
      ax[1].plot(force[:,4], label='wMy')
      ax[1].plot(force[:,5], label='wMz')
      #ax[2].plot(force[:,6], label='f1Fx')
      #ax[2].plot(force[:,7], label='f1Fy')
      #ax[2].plot(force[:,8], label='f1Fz')
      #ax[2].plot(force[:,9], label='f1Mx')
      #ax[2].plot(force[:,10], label='f1My')
      #ax[2].plot(force[:,11], label='f1Mz')
      #ax[3].plot(force[:,12], label='f1Fx')
      #ax[3].plot(force[:,13], label='f1Fy')
      #ax[3].plot(force[:,14], label='f1Fz')
      #ax[3].plot(force[:,15], label='f1Mx')
      #ax[3].plot(force[:,16], label='f1My')
      #ax[3].plot(force[:,17], label='f1Mz')

      plt.legend()
      plt.show()

    print(reward)
    #print(action_his_len)

  force_1 = 10
  force_2 = 30
  smooth_force = np.clip(force, -force_1, force_1) / force_1
  smooth_force_1 = np.clip(force , -force_2, force_2) / force_2
  smooth_force_2 = np.tanh(force * 0.1)

  fig, ax = plt.subplots(nrows=3, ncols=1, sharex=True)

  ax[0].title.set_text('Clip [-{}, {}]'.format(force_1, force_1))
  ax[0].plot(smooth_force[:,0], label='Fx')
  ax[0].plot(smooth_force[:,1], label='Fy')
  ax[0].plot(smooth_force[:,2], label='Fz')
  ax[0].plot(smooth_force[:,3], label='Mx')
  ax[0].plot(smooth_force[:,4], label='My')
  ax[0].plot(smooth_force[:,5], label='Mz')

  ax[1].title.set_text('Clip [-{}, {}]'.format(force_2, force_2))
  ax[1].plot(smooth_force_1[:,0], label='Fx')
  ax[1].plot(smooth_force_1[:,1], label='Fy')
  ax[1].plot(smooth_force_1[:,2], label='Fz')
  ax[1].plot(smooth_force_1[:,3], label='Mx')
  ax[1].plot(smooth_force_1[:,4], label='My')
  ax[1].plot(smooth_force_1[:,5], label='Mz')

  ax[2].title.set_text('tanh')
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
