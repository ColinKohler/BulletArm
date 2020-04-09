import os
import copy
import math
import numpy as np
import numpy.random as npr
from collections import deque

import pybullet as pb
import pybullet_data

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.robots.robot_base import RobotBase
import time

from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util
from helping_hands_rl_envs.simulators.pybullet.utils import object_generation
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class Kuka(RobotBase):
  '''

  '''
  def __init__(self):
    super().__init__()
    self.max_velocity = .35
    self.max_force = 200.
    self.finger_a_force = 2
    self.finger_b_force = 2
    self.finger_tip_force = 2
    self.end_effector_index = 14
    self.gripper_index = 7

    # lower limits for null space
    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    # upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    # joint ranges for null space
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    # restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    # joint damping coefficents
    self.jd = [
      0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
      0.00001, 0.00001, 0.00001, 0.00001
    ]

    self.home_positions = [0.3926, 0., -2.137, 1.432, 0, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0.]
    self.home_positions_joint = self.home_positions[:7]
    # self.home_positions = [
    #     0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
    #     -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
    # ]

    self.gripper_joint_limit = [0, 0.2]

  def initialize(self):
    ''''''
    ur5_urdf_filepath = os.path.join(self.root_dir, 'simulators/urdf/kuka/kuka_with_gripper2.sdf')
    self.id = pb.loadSDF(ur5_urdf_filepath)[0]
    pb.resetBasePositionAndOrientation(self.id, [-0.2,0,0], [0,0,0,1])

    # self.is_holding = False
    self.gripper_closed = False
    self.holding_obj = None
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]
    self.openGripper()

    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(7):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)

  def reset(self):
    self.gripper_closed = False
    self.holding_obj = None
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

  def closeGripper(self, max_it=100):
    ''''''
    p1, p2 = self._getGripperJointPosition()
    target = self.gripper_joint_limit[0]
    self._sendGripperCommand(target, target)
    self.gripper_closed = True
    it = 0
    while abs(target-p1) + abs(target-p2) > 0.001:
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getGripperJointPosition()
      if it > max_it or (abs(p1 - p1_) < 0.0001 and abs(p2 - p2_) < 0.0001):
        mean = (p1 + p2) / 2 - 0.01
        # self._sendGripperCommand(mean, mean)
        return False
      p1 = p1_
      p2 = p2_
    return True

  def adjustGripperCommand(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2) / 2 - 0.01
    self._sendGripperCommand(mean, mean)

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
    target = self.gripper_joint_limit[1]
    self._sendGripperCommand(target, target)
    self.gripper_closed = False
    self.holding_obj = None
    it = 0
    while abs(target-p1) + abs(target-p2) > 0.001:
      pb.stepSimulation()
      it += 1
      if it > 100:
        return False
      p1_, p2_ = self._getGripperJointPosition()
      if p1 >= p1_ and p2 >= p2_:
        return False
      p1 = p1_
      p2 = p2_
    return True

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:7]

  def _getGripperJointPosition(self):
    p1 = -pb.getJointState(self.id, 8)[0]
    p2 = pb.getJointState(self.id, 11)[0]
    return p1, p2

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                 targetVelocities=[0.]*num_motors,
                                 forces=[self.max_force]*num_motors,
                                 positionGains=[0.02]*num_motors,
                                 velocityGains=[1.0]*num_motors)
    # for i in range(num_motors):
    #   pb.setJointMotorControl2(bodyUniqueId=self.id,
    #                            jointIndex=i,
    #                            controlMode=pb.POSITION_CONTROL,
    #                            targetPosition=commands[i],
    #                            targetVelocity=0,
    #                            force=self.max_force,
    #                            maxVelocity=self.max_velocity,
    #                            positionGain=0.3,
    #                            velocityGain=1)

  def _sendGripperCommand(self, target_pos1, target_pos2):
    # pb.setJointMotorControl2(self.id,
    #                          7,
    #                          pb.POSITION_CONTROL,
    #                          targetPosition=target_pos,
    #                          force=self.max_force)
    pb.setJointMotorControl2(self.id,
                             8,
                             pb.POSITION_CONTROL,
                             targetPosition=-target_pos1,
                             force=self.finger_a_force)
    pb.setJointMotorControl2(self.id,
                             11,
                             pb.POSITION_CONTROL,
                             targetPosition=target_pos2,
                             force=self.finger_b_force)

    pb.setJointMotorControl2(self.id,
                             10,
                             pb.POSITION_CONTROL,
                             targetPosition=0,
                             force=self.finger_tip_force)
    pb.setJointMotorControl2(self.id,
                             13,
                             pb.POSITION_CONTROL,
                             targetPosition=0,
                             force=self.finger_tip_force)

