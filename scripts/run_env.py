import argparse
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import tqdm

from bulletarm import env_factory

def run(task, robot, plot_obs, render):
  workspace = np.array([[0.20, 0.60], [-0.2, 0.2], [-0.01, 0.25]])
  if 'close_loop' in task or 'force' in task:
    res = 64
    p = 12
    env_config = {
      'robot' : robot,
      'render' : render,
      'action_sequence' : 'pxyzr',
      'workspace' : workspace,
      'view_type' : 'camera_side_rgbd',
      'num_sensors' : 1,
      'num_objects' : 1,
      'physics_mode' : 'force',
      'max_steps' : 50,
      'obs_size' : res + p,
      'view_scale' : 1.0,
      'obs_type' : ['vision', 'force', 'proprio'],
      'random_orientation': True,
    }
    planner_config = {'dpos': 0.025, 'drot': np.pi/16}
  else:
    env_config = {'robot' : robot, 'render' : True}
    planner_config = {'half_rotation' : True}
  env = env_factory.createEnvs(0, task, env_config, planner_config)

  s = 0
  pbar = tqdm.tqdm(total=100)
  pbar.set_description('0/100')
  for n in range(100):
    obs = env.reset()
    done = False
    while not done:
      action = env.getNextAction()
      obs, reward, done = env.step(action)
      max_force = 100
      norm_force = np.clip(obs[1], -max_force, max_force) / max_force
      if plot_obs:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        #ax[0].imshow(obs[0][0].squeeze(), cmap='gray')
        ax[0].imshow(obs[0][:3][:,p//2:-p//2,p//2:-p//2].transpose(1,2,0))
        ax[1].plot(norm_force[:,0], label='Fx')
        ax[1].plot(norm_force[:,1], label='Fy')
        ax[1].plot(norm_force[:,2], label='Fz')
        ax[1].plot(norm_force[:,3], label='Mx')
        ax[1].plot(norm_force[:,4], label='My')
        ax[1].plot(norm_force[:,5], label='Mz')
        fig.legend()
        plt.show()
    s += reward
    pbar.set_description('{}/{}'.format(s, n+1))
    pbar.update(1)
  env.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to run')
  parser.add_argument('--robot', type=str, default='kuka',
    help='Robot to run')
  parser.add_argument('--plot_obs', default=False, action='store_true',
    help='Display observations.')
  parser.add_argument('--render', default=False, action='store_true',
    help='Render simulation.')

  args = parser.parse_args()
  run(args.task, args.robot, args.plot_obs, args.render)
