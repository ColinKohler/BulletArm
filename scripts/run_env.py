import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

from bulletarm import env_factory

def run(task, robot):
  workspace = np.array([[0.25, 0.65], [-0.2, 0.2], [0.01, 0.25]])
  if 'close_loop' in task or 'force' in task:
    env_config = {'robot' : robot, 'render' : True, 'action_sequence' : 'pxyzr', 'workspace' : workspace, 'view_type': 'render_center', 'physics_mode' : 'force', 'max_steps' : 50, 'obs_size' : 128, 'occlusion_prob' : 0.1, 'num_occlusions' : 0, 'view_scale' : 1.5}
    planner_config = {'dpos': 0.05, 'drot': np.pi/4}
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
      fig, ax = plt.subplots(nrows=1, ncols=1)
      #ax[0].plot(obs[3])
      ax.imshow(obs[2].squeeze(), cmap='gray')
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
