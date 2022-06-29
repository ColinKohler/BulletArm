import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_block_stacking_planner import CloseLoopBlockStackingPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException

class CloseLoopBlockStackingEnv(CloseLoopEnv):
  '''Close loop block stacking task.

  The robot needs to stack all N cubic blocks. The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    if 'num_objects' not in config:
      config['num_objects'] = 2
    super().__init__(config)
    assert self.num_obj >= 2

  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
      try:
        self._generateShapes(constants.CUBE, self.num_obj, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    return not self._isHolding() and self._checkStack(self.objects)

def createCloseLoopBlockStackingEnv(config):
  return CloseLoopBlockStackingEnv(config)
