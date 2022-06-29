import pybullet as pb
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.envs.close_loop_envs.close_loop_env import CloseLoopEnv
from bulletarm.pybullet.utils import transformations
from bulletarm.planners.close_loop_block_picking_corner_planner import CloseLoopBlockPickingCornerPlanner
from bulletarm.pybullet.equipments.corner import Corner

class CloseLoopBlockPickingCornerEnv(CloseLoopEnv):
  '''Close loop corner block picking task.

  The robot needs to slide the block away from the corner and then pick it up.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [1.2, 1.2]
    super().__init__(config)
    self.corner = Corner()
    self.corner_rz = 0
    self.corner_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]

  def resetCorner(self):
    self.corner_rz = np.random.random_sample() * 2*np.pi - np.pi if self.random_orientation else 0
    self.corner_pos = self._getValidPositions(0.2, 0, [], 1)[0]
    self.corner_pos.append(0)
    self.corner.reset(self.corner_pos, pb.getQuaternionFromEuler((0, 0, self.corner_rz)))

  def initialize(self):
    super().initialize()
    self.corner.initialize(pos=self.corner_pos)

  def reset(self):
    self.resetPybulletWorkspace()
    self.resetCorner()
    self.robot.moveTo([self.workspace[0].mean(), self.workspace[1].mean(), 0.2], transformations.quaternion_from_euler(0, 0, 0))
    pos, rot_q = self.corner.getObjPose()

    self._generateShapes(constants.CUBE, 1, pos=[pos], rot=[rot_q])
    return self._getObservation()

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., np.pi/2 * (np.random.random_sample() - 0.5)])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _checkTermination(self):
    gripper_z = self.robot._getEndEffectorPosition()[-1]
    return self.robot.holding_obj == self.objects[-1] and gripper_z > 0.08

  def getObjectPoses(self, objects=None):
    if objects is None: objects = self.objects + [self.corner]

    obj_poses = list()
    for obj in objects:
      pos, rot = obj.getPose()
      rot = self.convertQuaternionToEuler(rot)

      obj_poses.append(pos + rot)
    return np.array(obj_poses)

def createCloseLoopBlockPickingCornerEnv(config):
  return CloseLoopBlockPickingCornerEnv(config)
