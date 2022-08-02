import argparse
import numpy as np
import matplotlib.pyplot as plt

from bulletarm import env_factory

def runDemo(task, robot):
  if 'close_loop' in task or 'force' in task:
    env_config = {'robot' : robot, 'render' : True, 'action_sequence' : 'pxyzr', 'view_type': 'camera_center_xyz'}
  else:
    env_config = {'robot' : robot, 'render' : True}
  planner_config = {'dpos': 0.05, 'drot': np.pi/4}
  env = env_factory.createEnvs(0, task, env_config, planner_config)

  obs = env.reset()
  done = False
  while not done:
    action = env.getNextAction()
    obs, reward, done = env.step(action)
    breakpoint()
  env.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('task', type=str,
    help='Task to run')
  parser.add_argument('--robot', type=str, default='kuka',
    help='Robot to run')

  args = parser.parse_args()
  runDemo(args.task, args.robot)
