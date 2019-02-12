import time
from envs.pybullet_env import PyBulletEnv

env = PyBulletEnv()
env.reset()

while True:
  env.step(1)
