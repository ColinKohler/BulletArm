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

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(force[1:,0], label='Fx')
    ax.plot(force[1:,1], label='Fy')
    ax.plot(force[1:,2], label='Fz')
    ax.plot(force[1:,3], label='Mx')
    ax.plot(force[1:,4], label='My')
    ax.plot(force[1:,5], label='Mz')

    plt.legend()
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
