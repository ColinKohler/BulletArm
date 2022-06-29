import pybullet as pb
import numpy as np

from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_drawer_opening_planner import CloseLoopDrawerOpeningPlanner
from bulletarm.pybullet.equipments.drawer import Drawer

class CloseLoopDrawerOpeningEnv(CloseLoopEnv):
  '''Close loop drawer opening task.

  The robot needs to pull the handle of the drawer to open it.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    super().__init__(config)
    self.drawer = Drawer()
    self.drawer_rot = 0

  def initialize(self):
    super().initialize()
    self.drawer.initialize((self.workspace[0].mean(), self.workspace[1].mean(), 0), pb.getQuaternionFromEuler((0, 0, 0)), 0.3)

  def reset(self):
    self.resetPybulletWorkspace()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    # pos = self._getValidPositions(0.1, 0, [], 1)[0]
    # pos.append(0)
    pos = np.array([self.workspace[0].mean(), self.workspace[1].mean(), 0])
    self.drawer_rot = np.random.random()*2*np.pi if self.random_orientation else np.random.choice([np.pi/2, 3*np.pi/2])
    m = np.array(transformations.euler_matrix(0, 0, self.drawer_rot))[:3, :3]
    dx = np.random.random() * (0.2 - 0.15) + 0.15
    dy = np.random.random() * (0.1 - -0.1) + -0.1
    pos = pos + m[:, 0] * dx
    pos = pos + m[:, 1] * dy
    self.drawer.reset(pos, transformations.quaternion_from_euler(0, 0, self.drawer_rot))

    return self._getObservation()

  def _checkTermination(self):
    return self.drawer.isDrawerOpen()

  def getObjectPoses(self, objects=None):
    obj_poses = list()

    drawer_pos, drawer_rot = self.drawer.getPose()
    drawer_rot = self.convertQuaternionToEuler(drawer_rot)
    obj_poses.append(drawer_pos + drawer_rot)
    handle_pos = list(self.drawer.getHandlePosition())
    handle_rot = list(self.drawer.getHandleRotation())
    handle_rot = self.convertQuaternionToEuler(handle_rot)
    obj_poses.append(handle_pos + handle_rot)

    return np.array(obj_poses)

def createCloseLoopDrawerOpeningEnv(config):
  return CloseLoopDrawerOpeningEnv(config)
