import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_block_pulling_planner import CloseLoopBlockPullingPlanner
from bulletarm.pybullet.utils.constants import NoValidPositionException

class CloseLoopBlockPullingEnv(CloseLoopEnv):
  ''' Close loop block pulling task.

  The robot needs to pullone of the two blocks to make contact with the other block.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.8, 0.8]
    super().__init__(config)

  def reset(self):
    while True:
      self.resetPybulletWorkspace()
      self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
      try:
        if not self.random_orientation:
          padding = self._getDefaultBoarderPadding(constants.FLAT_BLOCK)
          min_distance = self._getDefaultMinDistance(constants.FLAT_BLOCK)
          x = np.random.random() * (self.workspace_size - padding) + self.workspace[0][0] + padding/2
          while True:
            y1 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding/2
            y2 = np.random.random() * (self.workspace_size - padding) + self.workspace[1][0] + padding/2
            if max(y1, y2) - min(y1, y2) > min_distance:
              break
          self._generateShapes(constants.FLAT_BLOCK, 2, pos=[[x, y1, self.object_init_z], [x, y2, self.object_init_z]], random_orientation=True)
        else:
          self._generateShapes(constants.FLAT_BLOCK, 2, random_orientation=self.random_orientation)
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
    return self.objects[0].isTouching(self.objects[1])

def createCloseLoopBlockPullingEnv(config):
  return CloseLoopBlockPullingEnv(config)
