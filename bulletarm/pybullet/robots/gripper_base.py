import os
import math
import numpy as np
import pybullet as pb
from scipy.ndimage import rotate

from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.robots.robot_base import RobotBase

class GripperBase(object):
  '''

  '''
  def __init__(self, finger_idxs, z_offset, joint_limit):
    super().__init__()

    self.finger_idxs = finger_idxs
    self.z_offset = z_offset
    self.joint_limit = joint_limit
    self.closed = False
    self.holding_obj = None

  def initialize(self, robot_id):
    self.robot_id = robot_id
    self.closed = False
    self.holding_obj = None

    self.open()

  def reset(self):
    self.closed = False
    self.holding_obj = None

    self.open()

  def open(self, max_it=100, force=10):
    ''''''
    self.closed = False
    self.holding_obj = None
    return self.control(1, max_it=max_it, force=force)

  def close(self, max_it=100, force=10):
    ''''''
    self.closed = True
    return self.control(0, max_it=max_it, force=force)

  def control(self, open_ratio, max_it=100, force=10):
    p1, p2 = self._getJointPosition()
    target = open_ratio * (self.joint_limit[1] - self.joint_limit[0]) + self.joint_limit[0]
    self._sendCommand(target, target, force)

    it = 0
    while abs(target - p1) + abs(target - p2) > 1e-3:
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getJointPosition()
      if it > max_it or (abs(p1 - p1_) < 1e-4 and abs(p2 - p2_) < 1e-4):
        return False
      p1 = p1_
      p2 = p2_
    return True

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
      finger_1_contact_points = pb.getContactPoints(self.robot_id, obj.object_id, self.finger_idxs[0])
      finger_2_contact_points = pb.getContactPoints(self.robot_id, obj.object_id, self.finger_idxs[1])
      finger_1_horizontal = list(filter(lambda p: abs(p[7][2]) < 0.3, finger_1_contact_points))
      finger_2_horizontal = list(filter(lambda p: abs(p[7][2]) < 0.3, finger_2_contact_points))
      if len(finger_1_horizontal) >= 1 and len(finger_2_horizontal) >=1:
        self.holding_obj = obj

  def _sendCommand(self, target_pos_1, target_pos_2, force=10):
    pb.setJointMotorControlArray(
      self.robot_id,
      [self.finger_idxs[0], self.finger_idxs[1]],
      pb.POSITION_CONTROL,
      [target_pos_1, target_pos_2],
      forces=[force, force]
    )

  def getOpenRatio(self):
    p1, p2 = self._getJointPosition()
    mean = (p1 + p2) / 2
    ratio = (mean - self.joint_limit[0]) / (self.joint_limit[1] - self.joint_limit[0])
    return ratio

  def adjustCommand(self):
    pass

  def checkClosed(self):
    limit = self.joint_limit[1]
    p1, p2 = self._getJointPosition()
    if (limit - p1) + (limit - p2) > 0.001:
      return
    else:
      self.holding_obj = None

  def _getJointPosition(self):
    p1 = pb.getJointState(self.robot_id, self.finger_idxs[0])[0]
    p2 = pb.getJointState(self.robot_id, self.finger_idxs[1])[0]
    return p1, p2
