import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.equipment.pivot import Pivot
from bulletarm.pybullet.utils import transformations

class CloseLoopPivotingEnv(CloseLoopEnv):
  '''Close loop pivoting task.

  The robot needs to slide the block away from the corner and then pick it up.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    super().__init__(config)
    self.pivot = Pivot()
    self.pivot_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    self.pivot_rz = 0

  def resetPivot(self):
    self.pivot_pos = self._getValidPositions(0.30, 0, [], 1)[0]
    self.pivot_pos.append(self.pivot.size[2] / 2)
    self.pivot_rz = np.random.random_sample() * 2*np.pi - np.pi if self.random_orientation else 0
    self.pivot.reset(self.pivot_pos, pb.getQuaternionFromEuler((0, 0, self.pivot_rz)))

  def initialize(self):
    super().initialize()
    self.pivot.initialize(pos=self.pivot_pos)

  def reset(self):
    self.resetPybulletWorkspace()
    self.resetPivot()

    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))

    pivoting_pos, pivoting_rot = self.pivot.getPivotingBlockPose()
    self.block = self._generateShapes(
      shape_type=constants.PIVOTING_BLOCK,
      pos=[pivoting_pos],
      rot=[pivoting_rot],
      wait=False
    )[0]
    return self._getObservation()

  def _checkTermination(self):
    return False
