import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations

class CloseLoopPivotingEnv(CloseLoopEnv):
  '''Close loop corner block picking task.

  The robot needs to slide the block away from the corner and then pick it up.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    super().__init__(config)

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))

    self.block = self._generateShapes(constants.FLAT_BLOCK, 1)[0]
    return self._getObservation()

  def _checkTermination(self):
    return False
