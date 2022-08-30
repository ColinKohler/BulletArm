import math
import pybullet as pb
from collections import namedtuple
from attrdict import AttrDict

JointInfo = namedtuple(
  "JointInfo",
   ["id", "name", "type", "lower_limit", "upper_limit", "max_force", "max_velocity"]
)
JOINT_TYPES = [
  "REVOLUTE",
  "PRISMATIC",
  "SPHERICAL",
  "PLANAR",
  "FIXED"
]

class Robotiq(object):
  ''' Robotiq 85f gripper. '''
  def __init__(self):
    self.end_effector_index = 12
    self.gripper_close_force = [30, 30]
    self.gripper_open_force = [30, 30]
    self.gripper_joint_limits = [0, 0.036]
    self.gripper_joint_names = list()
    self.gripper_joint_indices = list()
    self.position_gain = 0.01

    self.home_positions = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    self.robotiq_open_length_limit = [0, 0.085]
    self.robotiq_joint_limits = [
      0.715 - math.asin((self.robotiq_open_length_limit[0] - 0.010) / 0.1143),
      0.715 - math.asin((self.robotiq_open_length_limit[1] - 0.010) / 0.1143)
    ]

    self.robotiq_control_joints = [
      "robotiq_85_left_knuckle_joint",
       "robotiq_85_right_knuckle_joint",
       "robotiq_85_left_inner_knuckle_joint",
       "robotiq_85_right_inner_knuckle_joint",
       "robotiq_85_left_finger_tip_joint",
       "robotiq_85_right_finger_tip_joint"
    ]
    self.robotiq_main_control_joint = "robotiq_85_left_inner_knuckle_joint"
    self.robotiq_mimic_joints = [
      "robotiq_85_right_knuckle_joint",
      "robotiq_85_left_knuckle_joint",
      "robotiq_85_right_inner_knuckle_joint",
      "robotiq_85_left_finger_tip_joint",
      "robotiq_85_right_finger_tip_joint"
    ]
    self.robotiq_mimic_multiplier = [1, 1, 1, 1, -1, -1]
    self.robotiq_joints = AttrDict()

  def initialize(self, robot_id):
    self.id = robot_id
    self.num_joints = pb.getNumJoints(self.id)
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(0, 2):
        self.gripper_joint_names.append(str(joint_info[1]))
        self.gripper_joint_indices.append(i)
      elif i in range(4, self.num_joints):
        info = pb.getJointInfo(self.id, i)
        info = JointInfo(
          info[0],
          info[1].decode('utf-8'),
          JOINT_TYPES[info[2]],
          info[8],
          info[9],
          info[10],
          info[11])
        self.robotiq_joints[info.name] = info

  def enableFingerForceTorqueSensors(self):
    pass

  def getFingerForce(self):
    pass

  def controlGripper(self, open_ratio, max_it=100):
    p1, p2 = self._getGripperJointPosition()
    target = open_ratio * (self.gripper_joint_limits[0] - self.gripper_joint_limits[1]) + self.gripper_joint_limits[1]
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

  def openGripper(self):
    p1, p2 = self._getGripperJointPosition()
    limit = self.gripper_joint_limits[0]
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

  def closeGripper(self):
    p1, p2 = self._getGripperJointPosition()
    limit = self.gripper_joint_limits[1]
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

  def getGripperOpenRatio(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2)/2
    ratio = (mean - self.gripper_joint_limits[1]) / (self.gripper_joint_limits[0] - self.gripper_joint_limits[1])
    return ratio

  def getGripperImg(self, img_size, workspace_size, obs_size_m):
    pass

  def adjustGripperCommand(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2) / 2 + 0.005
    self._sendGripperCommand(mean, mean)

  def checkGripperClosed(self):
    pass

  def gripperHasForce(self):
    pass

  def _getGripperJointPosition(self):
    p1 = pb.getJointState(self.id, self.gripper_joint_indices[0])[0]
    p2 = pb.getJointState(self.id, self.gripper_joint_indices[1])[0]
    return p1, p2

  def _sendGripperCommand(self, target_pos1, target_pos2):
    pb.setJointMotorControlArray(
      self.id,
      self.gripper_joint_indices,
      pb.POSITION_CONTROL,
      targetPositions=[target_pos1, target_pos2],
      forces=self.gripper_open_force,
      positionGains=[self.position_gain] * 2,
      velocityGains=[1.0] * 2
    )

  def _setRobotiqPosition(self, pos):
    percentage = pos / self.gripper_joint_limits[1]
    target = percentage * (self.robotiq_joint_limits[0]-self.robotiq_joint_limits[1]) + self.robotiq_joint_limits[1]
    for i, jn in enumerate(self.robotiq_control_joints):
      motor = self.robotiq_joints[jn].id
      pb.resetJointState(self.id, motor, target*self.robotiq_mimic_multiplier[i])
      pb.setJointMotorControl2(
        self.id,
        motor,
        pb.POSITION_CONTROL,
        targetPosition=target*self.robotiq_mimic_multiplier[i],
        force=100
      )
