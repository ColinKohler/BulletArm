import os
import numpy as np
import pybullet as pb
from scipy.ndimage import rotate

from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.robots.robot_base import RobotBase
from bulletarm.pybullet.robots.gripper_base import GripperBase

class Kuka(RobotBase):
  '''

  '''
  def __init__(self):
    super().__init__()
    self.home_positions = [0.392, 0., -2.137, 1.432, 0, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0.]
    self.home_positions_joint = self.home_positions[:7]

    self.num_dofs = 7
    self.wrist_index = 7
    self.finger_idxs = [8, 11]
    self.end_effector_index = 14
    self.gripper_z_offset = 0.12
    self.gripper_joint_limit = [0, 0.2]
    self.adjust_gripper_offset = 0.01
    self.gripper = KukaGripper(self.finger_idxs, self.gripper_z_offset, self.gripper_joint_limit, self.adjust_gripper_offset)

    self.max_torque = [200.] * self.num_dofs
    #self.max_torque = [8.7, 8.7, 8.7, 8.7, 12.0, 12.0, 12.0]

    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    self.ml = [0, 0, 0, 1.575, 0, 0, 0]
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]

    self.urdf_filepath = os.path.join(constants.URDF_PATH, 'kuka/kuka_with_gripper2.sdf')

  def initialize(self):
    ''''''
    self.id = pb.loadSDF(self.urdf_filepath)[0]
    super().initialize()

class KukaGripper(GripperBase):
  '''

  '''
  def __init__(self, finger_idxs, z_offset, joint_limit, adjust_offset):
    super().__init__(finger_idxs, z_offset, joint_limit)
    self.adjust_offset = adjust_offset

  def _sendCommand(self, target_pos_1, target_pos_2, force=10):
    pb.setJointMotorControlArray(
      self.robot_id,
      [self.finger_idxs[0], self.finger_idxs[1]],
      pb.POSITION_CONTROL,
      [-target_pos_1, target_pos_2],
      forces=[force, force]
    )

  def _getJointPosition(self):
    p1 = pb.getJointState(self.robot_id, self.finger_idxs[0])[0]
    p2 = pb.getJointState(self.robot_id, self.finger_idxs[1])[0]
    return -p1, p2

  def adjustCommand(self):
    p1, p2 = self._getJointPosition()
    mean = (p1 + p2) / 2 - self.adjust_offset
    self._sendCommand(mean, mean)

  def getPickedObj(self, objects):
    '''
    Get the object which is currently being held by the gripper.

    Args:
      objects (numpy.array): Objects to check if are being held.

    Returns:
      (pybullet.objects.PybulletObject): Object being held.
    '''
    state = self.getOpenRatio()
    if not objects or state < 0.03:
      return None

    for obj in objects:
      # check the contact force normal to count the horizontal contact points
      finger_1_contact_points = pb.getContactPoints(self.robot_id, obj.object_id, 10)
      finger_2_contact_points = pb.getContactPoints(self.robot_id, obj.object_id, 13)
      finger_1_horizontal = list(filter(lambda p: abs(p[7][2]) < 0.3, finger_1_contact_points))
      finger_2_horizontal = list(filter(lambda p: abs(p[7][2]) < 0.3, finger_2_contact_points))
      if len(finger_1_horizontal) >= 1 and len(finger_2_horizontal) >=1:
        self.holding_obj = obj
