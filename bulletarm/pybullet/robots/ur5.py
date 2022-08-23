import os
import math
import pybullet as pb

from bulletarm.pybullet.robots.robot_base import RobotBase
from bulletarm.pybullet.robots.grippers.robotiq import Robotiq
from bulletarm.pybullet.robots.grippers.hydrostatic import Hydrostatic
from bulletarm.pybullet.robots.grippers.openhand_vf import OpenHandVF
from bulletarm.pybullet.utils import constants

COMPATABLE_GRIPPERS = ['robotiq', 'hydrostatic', 'openhand_vf']

class UR5(RobotBase):
  ''' UR5 robotic arm.

  This class implements robotic functions unique to the UR5 robotic arm. The gripper to be attached
  to the arm is specified by the argument.

  Args:
    gripper (string): The gripper type to be attached to the UR5.
  '''
  def __init__(self, gripper):
    super().__init__()

    # Setup gripper
    # NOTE: Might move this to the robot factory
    if gripper == 'robotiq':
      self.gripper = Robotiq()
      self.urdf_filepath = constants.UR5_ROBOTIQ_PATH
    elif gripper = 'hydrostatic':
      self.gripper = Hydrostatic()
      self.urdf_filepath = constants.UR5_HYDROSTATIC_PATH
    elif gripper = 'openhand_vf':
      self.gripper = OpenHandVF()
      self.urdf_filepath = constants.UR5_OPENHAND_VF_PATH

    # Setup arm
    self.end_effector_index = self.gripper.end_effector_index
    self.max_forces = [150, 150, 150, 28, 28, 28]
    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0.]

  def initialize(self):
    ''''''
    self.id = pb.loadURDF(self.urdf_filepath, [0,0,0.1], [0,0,0,1])
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

      elif i in range(14, self.num_joints):
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
        self.robotiq_joints[singleInfo.name] = singleInfo

  def reset(self):
    self.gripper_closed = False
    self.holding_obj = None
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

  def getGripperOpenRatio(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2)/2
    ratio = (mean - self.gripper_joint_limit[1]) / (self.gripper_joint_limit[0] - self.gripper_joint_limit[1])
    return ratio

  def controlGripper(self, open_ratio, max_it=100):
    p1, p2 = self._getGripperJointPosition()
    target = open_ratio * (self.gripper_joint_limit[0] - self.gripper_joint_limit[1]) + self.gripper_joint_limit[1]
    self._sendGripperCommand(target, target)
    it = 0
    while abs(target - p1) + abs(target - p2) > 0.001:
      self._setRobotiqPosition((p1 + p2) / 2)
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getGripperJointPosition()
      if it > max_it or (abs(p1 - p1_) < 0.0001 and abs(p2 - p2_) < 0.0001):
        return
      p1 = p1_
      p2 = p2_

  def closeGripper(self, max_it=100, primative=constants.PICK_PRIMATIVE):
    ''''''
    p1, p2 = self._getGripperJointPosition()
    limit = self.gripper_joint_limit[1]
    self._sendGripperCommand(limit, limit)
    # self._sendGripperCloseCommand()
    self.gripper_closed = True
    it = 0
    while (limit-p1) + (limit-p2) > 0.001:
      self._setRobotiqPosition((p1 + p2) / 2)
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getGripperJointPosition()
      if it > max_it or (abs(p1-p1_)<0.0001 and abs(p2-p2_)<0.0001):
        mean = (p1+p2)/2 + 0.005
        self._sendGripperCommand(mean, mean)
        return False
      p1 = p1_
      p2 = p2_
    return True

  def adjustGripperCommand(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2) / 2 + 0.005
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
    limit = self.gripper_joint_limit[0]
    self._sendGripperCommand(limit, limit)
    self.gripper_closed = False
    it = 0
    while p1 > 0.0:
      self._setRobotiqPosition((p1 + p2) / 2)
      pb.stepSimulation()
      it += 1
      if it > 100:
        return False
      p1, p2 = self._getGripperJointPosition()
    return True

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:-8]

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                 [0.]*num_motors, self.max_forces[:-2], [0.02]*num_motors, [1.0]*num_motors)
