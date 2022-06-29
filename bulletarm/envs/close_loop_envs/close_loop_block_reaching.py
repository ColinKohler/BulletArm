import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_block_picking_planner import CloseLoopBlockPickingPlanner

class CloseLoopBlockReachingEnv(CloseLoopEnv):
  ''' Close loop block reaching task.

  The robot needs to place the gripper close to a cubic block.

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

  def _checkTermination(self):
    gripper_pos = self.robot._getEndEffectorPosition()
    obj_pos = self.objects[0].getPosition()
    return np.linalg.norm(np.array(gripper_pos) - np.array(obj_pos)) < 0.03

def createCloseLoopBlockReachingEnv(config):
  return CloseLoopBlockReachingEnv(config)
