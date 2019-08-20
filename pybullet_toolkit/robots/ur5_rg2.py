import os
import copy
import numpy as np
import numpy.random as npr
from collections import deque

import pybullet as pb
import pybullet_data

import helping_hands_rl_envs
import time
from helping_hands_rl_envs.pybullet_toolkit.robots.robot_base import RobotBase

from helping_hands_rl_envs.pybullet_toolkit.utils import pybullet_util
from helping_hands_rl_envs.pybullet_toolkit.utils import object_generation
from helping_hands_rl_envs.pybullet_toolkit.utils import transformations

class UR5_RG2(RobotBase):
  '''

  '''
  def __init__(self):
    super().__init__()
    # Setup arm and gripper variables
    self.max_forces = [150, 150, 150, 28, 28, 28, 30, 30]
    self.gripper_close_force = [30] * 2
    self.gripper_open_force = [30] * 2
    self.end_effector_index = 12

    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0.]
    self.home_positions_joint = self.home_positions[1:7]
    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)

    self.gripper_joint_limit = [0, 0.036]
    self.gripper_joint_names = list()
    self.gripper_joint_indices = list()

  def reset(self):
    ''''''
    ur5_urdf_filepath = os.path.join(self.root_dir, 'urdf/ur5/ur5_w_simple_gripper.urdf')
    self.id = pb.loadURDF(ur5_urdf_filepath, [0,0,0], [0,0,0,1])
    # self.is_holding = False
    self.gripper_closed = False
    self.holding_obj = None
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    self.gripper_joint_names = list()
    self.gripper_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(1, 7):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)
      elif i in range(10, 12):
        self.gripper_joint_names.append(str(joint_info[1]))
        self.gripper_joint_indices.append(i)

  def closeGripper(self, max_it=100):
    ''''''
    p1, p2 = self._getGripperJointPosition()
    limit = self.gripper_joint_limit[1]
    self._sendGripperCloseCommand()
    self.gripper_closed = True
    it = 0
    while (limit-p1) + (limit-p2) > 0.001:
    # while p1 < 0.036:
      pb.stepSimulation()
      it += 1
      if it > max_it:
        self._sendGripperCommand(p1+0.01, p2+0.01)
        return False
      p1_, p2_ = self._getGripperJointPosition()
      if p1 >= p1_ and p2 >= p2_:
        self._sendGripperCommand(p1+0.01, p2+0.01)
        return False
      p1 = p1_
      p2 = p2_
    return True

  def checkGripperClosed(self):
    limit = self.gripper_joint_limit[1]
    p1, p2 = self._getGripperJointPosition()
    if (limit - p1) + (limit - p2) > 0.001:
      return
    else:
      self.holding_obj = None

  def openGripper(self):
    ''''''
    p1, p2 = self._getGripperJointPosition()
    self._sendGripperOpenCommand()
    self.gripper_closed = False
    self.holding_obj = None
    it = 0
    while p1 > 0.0:
      pb.stepSimulation()
      it += 1
      if it > 100:
        return False
      p1, p2 = self._getGripperJointPosition()
    return True

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:-2]

  def _getGripperJointPosition(self):
    p1 = pb.getJointState(self.id, self.gripper_joint_indices[0])[0]
    p2 = pb.getJointState(self.id, self.gripper_joint_indices[1])[0]
    return p1, p2

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                 [0.]*num_motors, self.max_forces[:-2], [0.02]*num_motors, [1.0]*num_motors)

  def _sendGripperCloseCommand(self):
    target_pos = self.gripper_joint_limit[1] + 0.01
    pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
                                 targetPositions=[target_pos, target_pos], forces=self.gripper_close_force)

  def _sendGripperOpenCommand(self):
    target_pos = self.gripper_joint_limit[0] - 0.01
    pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
                                 targetPositions=[target_pos, target_pos], forces=self.gripper_open_force)

  def _sendGripperCommand(self, target_pos1, target_pos2):
    pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
                                 targetPositions=[target_pos1, target_pos2], forces=self.gripper_open_force)