import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_house_building_1_planner import CloseLoopHouseBuilding1Planner
from bulletarm.pybullet.utils.constants import NoValidPositionException

class CloseLoopHouseBuilding1Env(CloseLoopEnv):
  '''Close loop house building 1 task.

  The robot needs to stack N-1 cubic blocks then place a triangle block on top of the stack.
  The number of blocks N is set by the config.

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
        self._generateShapes(constants.TRIANGLE, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.CUBE, self.num_obj-1, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return not self._isHolding() and self._checkStack(blocks + triangles) and self._checkObjUpright(triangles[0]) and self._isObjOnTop(triangles[0])

def createCloseLoopHouseBuilding1Env(config):
  return CloseLoopHouseBuilding1Env(config)
