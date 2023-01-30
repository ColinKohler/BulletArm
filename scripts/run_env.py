import argparse
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from bulletarm import env_factory

def run(task, robot, plot_obs):
  workspace = np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]])
  if 'close_loop' in task or 'force' in task:
    env_config = {
      'robot' : robot,
      'render' : True,
      'action_sequence' : 'pxyzr',
      'workspace' : workspace,
      'view_type' : 'camera_side_rgbd',
      'num_sensors' : 1,
      'physics_mode' : 'force',
      'max_steps' : 50,
      'obs_size' : 74,
      'view_scale' : 1.0,
      'obs_type' : ['depth', 'force', 'proprio']
    }
    planner_config = {'dpos': 0.025, 'drot': np.pi/16}
  else:
    env_config = {'robot' : robot, 'render' : True}
    planner_config = {'half_rotation' : True}
  env = env_factory.createEnvs(0, task, env_config, planner_config)

  for _ in range(20):
    obs = env.reset()
    done = False
    while not done:
      action = env.getNextAction()
      obs, reward, done = env.step(action)
      norm_force = np.clip(obs[1], -10, 10) / 10
      if plot_obs:
        fig, ax = plt.subplots(nrows=1, ncols=3)
        ax[0].imshow(obs[0][3].squeeze(), cmap='gray')
        ax[1].imshow(obs[0][:3][:,6:-6,6:-6].transpose(1,2,0))
        ax[2].plot(obs[1][:,0], label='Fx')
        ax[2].plot(obs[1][:,1], label='Fy')
        ax[2].plot(obs[1][:,2], label='Fz')
        ax[2].plot(obs[1][:,3], label='Mx')
        ax[2].plot(obs[1][:,4], label='My')
        ax[2].plot(obs[1][:,5], label='Mz')
        fig.legend()
        plt.show()
      print(reward)
  env.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to run')
  parser.add_argument('--robot', type=str, default='kuka',
    help='Robot to run')
  parser.add_argument('--plot_obs', default=False, action='store_true',
    help='Display observations.')

  args = parser.parse_args()
  run(args.task, args.robot, args.plot_obs)
