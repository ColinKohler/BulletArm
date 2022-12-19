import argparse
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from bulletarm import env_factory

def run(task, robot):
  workspace = np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]])
  if 'close_loop' in task or 'force' in task:
    env_config = {'robot' : robot, 'render' : True, 'action_sequence' : 'pxyzr', 'workspace' : workspace, 'view_type': 'render_center', 'physics_mode' : 'force', 'max_steps' : 50, 'obs_size' : 128, 'occlusion_prob' : 0.1, 'num_occlusions' : 0, 'view_scale' : 1.5, 'obs_type' : 'pixel+force+proprio'}
    planner_config = {'dpos': 0.025, 'drot': np.pi/16}
  else:
    env_config = {'robot' : robot, 'render' : True}
    planner_config = {'half_rotation' : True}
  env = env_factory.createEnvs(0, task, env_config, planner_config)

  for _ in range(20):
    obs = env.reset()
    done = False
    i = 0
    while not done:
      action = env.getNextAction()
      obs, reward, done = env.step(action)
      norm_force = np.clip(obs[3], -10, 10) / 10
      #print(np.max(np.abs(norm_force)))
      #print(np.mean(np.abs(norm_force)))
      #print(np.mean(np.abs(norm_force)) > 18e-3)
      if i >= 1000:
        fig, ax = plt.subplots(nrows=1, ncols=2)
        ax[0].plot(obs[3][:,0], label='Fx')
        ax[0].plot(obs[3][:,1], label='Fy')
        ax[0].plot(obs[3][:,2], label='Fz')
        ax[0].plot(obs[3][:,3], label='Mx')
        ax[0].plot(obs[3][:,4], label='My')
        ax[0].plot(obs[3][:,5], label='Mz')
        ax[1].imshow(obs[2].squeeze(), cmap='gray')
        fig.legend()
        plt.show()
      i += 1
  env.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to run')
  parser.add_argument('--robot', type=str, default='kuka',
    help='Robot to run')

  args = parser.parse_args()
  run(args.task, args.robot)
