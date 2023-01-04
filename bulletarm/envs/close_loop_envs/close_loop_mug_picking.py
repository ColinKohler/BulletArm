import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.equipment.tray import Tray

class CloseLoopMugPickingEnv(CloseLoopEnv):
  ''' Close loop mug picking task.

  The robot needs to pick the mug by the handle.

  Args:
    config (dict): Intialization arguments for the env
  '''

  def __init__(self, config):
    super().__init__(config)

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    self._generateShapes(constants.MUG, 1, random_orientation=self.random_orientation, padding=0)
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    gripper_pos = self.robot._getEndEffectorPosition()
    mug_grasp_pos = self.objects[-1].getGraspPosition()

    return (self.robot.getHeldObject() == self.objects[-1] and
            np.allclose(gripper_pos[:2], mug_grasp_pos[:2], atol=1e-2) and
            gripper_pos[-1] > 0.15)
