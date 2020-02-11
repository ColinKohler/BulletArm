import os
import copy
import math
import numpy as np
import numpy.random as npr
from collections import deque, namedtuple
from attrdict import AttrDict
from threading import Thread

import pybullet as pb
import pybullet_data

import helping_hands_rl_envs
import time
from helping_hands_rl_envs.simulators.pybullet.robots.robot_base import RobotBase

from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util
from helping_hands_rl_envs.simulators.pybullet.utils import object_generation
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

jointInfo = namedtuple("jointInfo",
                       ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity"])
jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]

class UR5_Robotiq(RobotBase):
  def __init__(self):
    super(UR5_Robotiq, self).__init__()
    # Setup arm and gripper variables
    self.max_forces = [150, 150, 150, 28, 28, 28, 30, 30]
    self.gripper_close_force = [30] * 2
    self.gripper_open_force = [30] * 2
    self.end_effector_index = 18

    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    self.home_positions_joint = self.home_positions[1:7]
    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)

    # the open length of the gripper. 0 is closed, 0.085 is completely opened
    self.gripper_open_length_limit = [0, 0.085]
    # the corresponding robotiq_85_left_knuckle_joint limit
    self.gripper_joint_limit = [0.715 - math.asin((self.gripper_open_length_limit[0] - 0.010) / 0.1143),
                                0.715 - math.asin((self.gripper_open_length_limit[1] - 0.010) / 0.1143)]

    self.controlJoints = ["robotiq_85_left_knuckle_joint",
                     "robotiq_85_right_knuckle_joint",
                     "robotiq_85_left_inner_knuckle_joint",
                     "robotiq_85_right_inner_knuckle_joint",
                     "robotiq_85_left_finger_tip_joint",
                     "robotiq_85_right_finger_tip_joint"]
    self.gripper_main_control_joint_name = "robotiq_85_left_inner_knuckle_joint"
    self.gripper_mimic_joint_name = [
      "robotiq_85_right_knuckle_joint",
      "robotiq_85_left_knuckle_joint",
      "robotiq_85_right_inner_knuckle_joint",
      "robotiq_85_left_finger_tip_joint",
      "robotiq_85_right_finger_tip_joint"
    ]
    self.gripper_mimic_multiplier = [1, 1, 1, -1, -1]
    self.gripper_joints = AttrDict()


    self.holding_obj = None
    self.gripper_closed = False
    self.state = {
      'holding_obj': self.holding_obj,
      'gripper_closed': self.gripper_closed
    }


  def reset(self):
    ''''''
    ur5_urdf_filepath = os.path.join(self.root_dir, 'simulators/urdf/ur5/ur5_robotiq_85_gripper.urdf')
    self.id = pb.loadURDF(ur5_urdf_filepath, [0,0,0], [0,0,0,1])
    # self.is_holding = False
    self.gripper_closed = False
    self.holding_obj = None
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(1, 7):
        self.arm_joint_names.append(joint_info[1].decode('UTF-8'))
        self.arm_joint_indices.append(i)

      elif i in range(10, self.num_joints):
        info = pb.getJointInfo(self.id, i)
        jointID = info[0]
        jointName = info[1].decode("utf-8")
        jointType = jointTypeList[info[2]]
        jointLowerLimit = info[8]
        jointUpperLimit = info[9]
        jointMaxForce = info[10]
        jointMaxVelocity = info[11]
        singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce,
                               jointMaxVelocity)
        self.gripper_joints[singleInfo.name] = singleInfo
    self.gripper_joint_indices = [self.gripper_joints['robotiq_85_left_knuckle_joint'].id,
                                  self.gripper_joints['robotiq_85_right_knuckle_joint'].id]

  def closeGripper(self, max_it=1000):
    ''''''
    self._sendGripperCommand(self.gripper_joint_limit[0])
    target = self.gripper_joint_limit[0]
    p1, p2 = self._getGripperJointPosition()
    it = 0
    while abs(target-p1) + abs(target-p2) > 0.001:
      pb.stepSimulation()
      it += 1
      if it > max_it:
        return False

      f1, f2 = self._getGripperJointForce()
      if f1 >= 1 and f2 >= 1:
        self._sendGripperCommand(p1)
        self._sendTipCommand(p1-0.01)
        return False

      p1, p2 = self._getGripperJointPosition()
    return True

  def checkGripperClosed(self):
    target = self.gripper_joint_limit[0]
    p1, p2 = self._getGripperJointPosition()
    if abs(target-p1) + abs(target-p2) > 0.001:
      return
    else:
      self.holding_obj = None

  def openGripper(self):
    ''''''
    self._sendGripperCommand(self.gripper_joint_limit[1])
    target = self.gripper_joint_limit[1]
    p1, p2 = self._getGripperJointPosition()
    it = 0
    while abs(target - p1) + abs(target - p2) > 0.001:
      pb.stepSimulation()
      it += 1
      if it > 100:
        return False
      p1, p2 = self._getGripperJointPosition()
    return True

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:6]

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                 [0.]*num_motors, self.max_forces[:-2], [0.02]*num_motors, [1.0]*num_motors)

  def _getGripperJointPosition(self):
    p1 = pb.getJointState(self.id, self.gripper_joint_indices[0])[0]
    p2 = pb.getJointState(self.id, self.gripper_joint_indices[1])[0]
    return p1, p2

  def _getGripperJointForce(self):
    f1 = pb.getJointState(self.id, self.gripper_joints['robotiq_85_left_finger_tip_joint'].id)[3]
    f2 = pb.getJointState(self.id, self.gripper_joints['robotiq_85_right_finger_tip_joint'].id)[3]
    return f1, f2

  def _sendGripperCommand(self, target):
    pb.setJointMotorControl2(self.id,
                             self.gripper_joints[self.gripper_main_control_joint_name].id,
                             pb.POSITION_CONTROL,
                             targetPosition=target,
                             force=self.gripper_joints[self.gripper_main_control_joint_name].maxForce,
                             positionGain=0.02,
                             velocityGain=1.0)
    for i in range(len(self.gripper_mimic_joint_name)):
      joint = self.gripper_joints[self.gripper_mimic_joint_name[i]]
      pb.setJointMotorControl2(self.id, joint.id, pb.POSITION_CONTROL,
                               targetPosition=target * self.gripper_mimic_multiplier[i],
                               force=joint.maxForce,
                               positionGain=0.02,
                               velocityGain=1.0)
  def _sendTipCommand(self, target):
    pb.setJointMotorControl2(self.id,
                             self.gripper_joints['robotiq_85_left_finger_tip_joint'].id,
                             pb.POSITION_CONTROL,
                             targetPosition=target * self.gripper_mimic_multiplier[-2],
                             force=self.gripper_joints['robotiq_85_left_finger_tip_joint'].maxForce,
                             positionGain=0.02,
                             velocityGain=1.0)
    pb.setJointMotorControl2(self.id,
                             self.gripper_joints['robotiq_85_right_finger_tip_joint'].id,
                             pb.POSITION_CONTROL,
                             targetPosition=target * self.gripper_mimic_multiplier[-1],
                             force=self.gripper_joints['robotiq_85_right_finger_tip_joint'].maxForce,
                             positionGain=0.02,
                             velocityGain=1.0)