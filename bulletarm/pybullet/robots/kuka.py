import os
import math
import pybullet as pb
from scipy.ndimage import rotate

from bulletarm.pybullet.robots.robot_base import RobotBase
from bulletarm.pybullet.utils import constants

class Kuka(RobotBase):
  ''' Kuka robotic arm.

  This class implements robotic functions unique to the Kuka robotic arm.
  '''
  def __init__(self):
    super().__init__()

    self.urdf_filepath = constants.KUKA_PATH

    # Setup arm
    self.num_dofs = 7
    self.max_forces = [200.] * self.num_dofs
    self.max_velocities = [.35] * self.num_dofs

    self.lower_limits = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    self.upper_limts = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    self.joint_ranges = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    self.rest_poses = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    self.joint_damping = [
      0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
      0.00001, 0.00001, 0.00001, 0.00001
    ]
    self.home_positions = [0.3926, 0., -2.137, 1.432, 0, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0.]

    # Setup gripper
    self.end_effector_index = 14
    self.wrist_index = 7
    self.finger_a_index = 8
    self.finger_b_index = 11
    self.gripper_z_offset = 0.12
    self.gripper_joint_limit = [0, 0.2]
    self.adjust_gripper_offset = 0.01

  def initialize(self):
    ''''''
    self.id = pb.loadSDF(self.urdf_filepath, useFixedBase=True)[0]
    pb.resetBasePositionAndOrientation(self.id, [-0.1,0,0], [0,0,0,1])

    self.gripper_closed = False
    self.holding_obj = None

    # Enable force sensors
    pb.enableJointForceTorqueSensor(self.id, self.wrist_index)
    pb.enableJointForceTorqueSensor(self.id, self.finger_a_index)
    pb.enableJointForceTorqueSensor(self.id, self.finger_b_index)

    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(self.num_dofs):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)

    # Zero force out
    self.force_history = list()
    pb.stepSimulation()
    force, moment = self.getWristForce()
    self.zero_force = np.concatenate((force, moment))

    self.openGripper()

  def reset(self):
    self.gripper_closed = False
    self.holding_obj = None

    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]
    self.openGripper()

    # Zero force out
    self.force_history = list()
    pb.stepSimulation()
    force, moment = self.getWristForce()
    self.zero_force = np.concatenate((force, moment))

  def getWristForce(self):
    wrist_info = list(pb.getJointState(self.id, self.wrist_index)[2])
    wrist_force = np.array(wrist_info[:3])
    wrist_moment = np.array(wrist_info[3:])

    # Transform to world frame
    wrist_rot = pb.getMatrixFromQuaternion(pb.getLinkState(self.id, self.wrist_index - 1)[5])
    wrist_rot = np.array(list(wrist_rot)).reshape((3,3))
    wrist_force = np.dot(wrist_rot, wrist_force)
    wrist_moment = np.dot(wrist_rot, wrist_moment)

    return wrist_force, wrist_moment

  def getFingerForce(self):
    finger_a_force = pb.getJointState(self.id, 8)[2]
    finger_b_force = pb.getJointState(self.id, 11)[2]

    return finger_a_force, finger_b_force

  def controlGripper(self, open_ratio, max_it=100):
    p1, p2 = self._getGripperJointPosition()
    target = open_ratio * (self.gripper_joint_limit[1] - self.gripper_joint_limit[0]) + self.gripper_joint_limit[0]
    self._sendGripperCommand(target, target)
    it = 0
    while abs(target - p1) + abs(target - p2) > 0.001:
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getGripperJointPosition()
      if it > max_it or (abs(p1 - p1_) < 0.0001 and abs(p2 - p2_) < 0.0001):
        return
      p1 = p1_
      p2 = p2_

  def openGripper(self):
    ''''''
    p1, p2 = self._getGripperJointPosition()
    target = self.gripper_joint_limit[1]
    self._sendGripperCommand(target, target)
    self.gripper_closed = False
    self.holding_obj = None
    it = 0
    if self.holding_obj:
      pos, rot = self.holding_obj.getPose()
    while abs(target-p1) + abs(target-p2) > 0.001:
      if self.holding_obj and it < 5:
        self.holding_obj.resetPose(pos, rot)
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

  def closeGripper(self, max_it=100, primative=constants.PICK_PRIMATIVE):
    ''''''
    if primative == constants.PULL_PRIMATIVE:
      force = 20
    else:
      force = 2
    p1, p2 = self._getGripperJointPosition()
    target = self.gripper_joint_limit[0]
    self._sendGripperCommand(target, target, force)
    self.gripper_closed = True
    it = 0
    while abs(target-p1) + abs(target-p2) > 0.001:
      pb.stepSimulation()
      it += 1
      p1_, p2_ = self._getGripperJointPosition()
      if it > max_it or (abs(p1 - p1_) < 0.0001 and abs(p2 - p2_) < 0.0001):
        return False
      p1 = p1_
      p2 = p2_
    return True

  def getGripperOpenRatio(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2)/2
    ratio = (mean - self.gripper_joint_limit[0]) / (self.gripper_joint_limit[1] - self.gripper_joint_limit[0])
    return ratio

  def getGripperImg(self, img_size, workspace_size, obs_size_m):
    gripper_state = self.getGripperOpenRatio()
    gripper_rz = pb.getEulerFromQuaternion(self._getEndEffectorRotation())[2]

    im = np.zeros((img_size, img_size))
    gripper_half_size = 4 * workspace_size / obs_size_m
    gripper_half_size = round(gripper_half_size / 128 * img_size)
    gripper_max_open = 35 * workspace_size / obs_size_m

    anchor = img_size // 2
    d = int(gripper_max_open / 128 * img_size * gripper_state)
    im[int(anchor - d // 2 - gripper_half_size):int(anchor - d // 2 + gripper_half_size), int(anchor - gripper_half_size):int(anchor + gripper_half_size)] = 1
    im[int(anchor + d // 2 - gripper_half_size):int(anchor + d // 2 + gripper_half_size), int(anchor - gripper_half_size):int(anchor + gripper_half_size)] = 1
    im = rotate(im, np.rad2deg(gripper_rz), reshape=False, order=0)

    return im

  def adjustGripperCommand(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2) / 2 - self.adjust_gripper_offset
    self._sendGripperCommand(mean, mean)

  def checkGripperClosed(self):
    limit = self.gripper_joint_limit[1]
    p1, p2 = self._getGripperJointPosition()
    if (limit - p1) + (limit - p2) > 0.001:
      return
    else:
      self.holding_obj = None

  def gripperHasForce(self):
    return (pb.getJointState(self.id, self.finger_a_index)[3] >= 2 or
            pb.getJointState(self.id, self.finger_b_index)[3] <= -2

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(
      self.id,
      self.end_effector_index,
      pos,
      targetOrientation=rot,
      jointDamping=self.joint_damping)[:self.num_dofs]

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
                                 positionGains=[self.position_gain]*num_motors,
                                 velocityGains=[1.0]*num_motors)

  def _sendGripperCommand(self, target_pos1, target_pos2, force=2):
    pb.setJointMotorControlArray(self.id,
                                 [8, 11, 10, 13],
                                 pb.POSITION_CONTROL,
                                 [-target_pos1, target_pos2, 0, 0],
                                 forces=[force, force,  force, force])
