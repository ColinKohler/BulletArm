import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_block_picking_planner import CloseLoopBlockPickingPlanner
from bulletarm.pybullet.equipments.tray import Tray

class CloseLoopBlockPickingEnv(CloseLoopEnv):
  ''' Close loop block picking task.

  The robot needs to pick up all N cubic blocks. The number of blocks N is set by the config.

  Args:
    config (dict): Intialization arguments for the env
  '''

  def __init__(self, config):
    super().__init__(config)

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self._generateShapes(constants.CUBE, 1, random_orientation=self.random_orientation)
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    gripper_z = self.robot._getEndEffectorPosition()[-1]
    return self.robot.holding_obj == self.objects[-1] and gripper_z > 0.15

def createCloseLoopBlockPickingEnv(config):
  return CloseLoopBlockPickingEnv(config)
