import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.friction_fingers.friction_fingers_env import FrictionFingersEnv

class FrictionFingersObjectShiftingEnv(FrictionFingersEnv):
  ''' Friction Fingers Object Shifting task.

  The robot
  The robot needs to pick up all N cubic blocks. The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''

  def __init__(self, config):
    super().__init__(config)

  def reset(self):
    self.resetPybulletWorkspace()
    self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
    return self._getObservation()

  def _checkTermination(self):
    return False

if __name__ == '__main__':
  env_config = {'render' : True, 'seed' : 0}
  env = FrictionFingersObjectShiftingEnv(env_config)
  obs = env.reset()

  done = False
  while not done:
    pass
