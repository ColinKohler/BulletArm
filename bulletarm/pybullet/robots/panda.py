import os
import math
import numpy as np
import pybullet as pb
from scipy.ndimage import rotate
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.robots.robot_base import RobotBase

class Panda(RobotBase):
  '''

  '''
  def __init__(self):
    super().__init__()
    self.home_positions = [-0.60, -0.14, 0.59, -2.40, 0.11, 2.28, -1, 0.0, 0, 0, 0, 0, 0, 0, 0]
    self.home_positions_joint = self.home_positions[:7]
    #self.max_force = 240
    self.max_forces = [150, 150, 150, 30, 30, 30, 30]
    self.position_gain = 1.0

    self.num_dofs = 7
    self.wrist_index = 8
    self.finger_a_index = 10
    self.finger_b_index = 12
    self.end_effector_index = 13
    self.gripper_z_offset = 0.08
    self.gripper_joint_limit = [0, 0.04]

    self.ll = [-7]*self.num_dofs
    self.ul = [7]*self.num_dofs
    self.jr = [7]*self.num_dofs

    self.urdf_filepath = os.path.join(constants.URDF_PATH, 'franka_panda/panda.urdf')

  def initialize(self):
    ''''''
    self.id = pb.loadURDF(self.urdf_filepath, useFixedBase=True)
    pb.resetBasePositionAndOrientation(self.id, [-0.1,0,0], [0,0,0,1])

    self.gripper_closed = False
    self.holding_obj = None

    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    pb.enableJointForceTorqueSensor(self.id, self.wrist_index)
    pb.enableJointForceTorqueSensor(self.id, self.finger_a_index)
    pb.enableJointForceTorqueSensor(self.id, self.finger_b_index)

    c = pb.createConstraint(self.id,
                            self.finger_a_index,
                            self.id,
                            self.finger_b_index,
                            jointType=pb.JOINT_GEAR,
                            jointAxis=[1, 0, 0],
                            parentFramePosition=[0, 0, 0],
                            childFramePosition=[0, 0, 0])
    pb.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)

    for j in range(pb.getNumJoints(self.id)):
      pb.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)

    self.openGripper()

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


  def reset(self):
    self.gripper_closed = False
    self.holding_obj = None

    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]
    self.moveToJ(self.home_positions_joint[:self.num_dofs])
    self.openGripper()

    # Zero force out
    self.force_history = list()
    force, moment = self.getWristForce()
    self.zero_force = np.concatenate((force, moment))

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

  def getGripperOpenRatio(self):
    p1, p2 = self._getGripperJointPosition()
    mean = (p1 + p2) / 2
    ratio = (mean - self.gripper_joint_limit[0]) / (self.gripper_joint_limit[1] - self.gripper_joint_limit[0])
    return ratio

  def closeGripper(self, max_it=100, primative=constants.PICK_PRIMATIVE):
    ''''''
    if primative == constants.PULL_PRIMATIVE:
      force = 20
    else:
      force = 10
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
        # mean = (p1 + p2) / 2 - 0.001
        # self._sendGripperCommand(mean, mean)
        return False
      p1 = p1_
      p2 = p2_
    return True

  def adjustGripperCommand(self):
    pass

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

  def gripperHasForce(self):
    return (pb.getJointState(self.id, self.finger_a_index)[3] >= 2 or
            pb.getJointState(self.id, self.finger_b_index)[3] <= -2)


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
    finger_a_info = list(pb.getJointState(self.id, self.finger_a_index)[2])
    finger_a_force = np.array(finger_a_info[:3])
    finger_a_moment = np.array(finger_a_info[3:])

    finger_b_info = list(pb.getJointState(self.id, self.finger_b_index)[2])
    finger_b_force = np.array(finger_b_info[:3])
    finger_b_moment = np.array(finger_b_info[3:])

    # Transform to world frame
    finger_a_rot = pb.getMatrixFromQuaternion(pb.getLinkState(self.id, self.finger_a_index - 1)[5])
    finger_a_rot = np.array(list(finger_a_rot)).reshape((3,3))
    finger_a_force = np.dot(finger_a_rot, finger_a_force)
    finger_a_moment = np.dot(finger_a_rot, finger_a_moment)

    finger_b_rot = pb.getMatrixFromQuaternion(pb.getLinkState(self.id, self.finger_b_index - 1)[5])
    finger_b_rot = np.array(list(finger_b_rot)).reshape((3,3))
    finger_b_force = np.dot(finger_b_rot, finger_b_force)
    finger_b_moment = np.dot(finger_b_rot, finger_b_moment)

    return finger_a_force, finger_a_moment, finger_b_force, finger_b_moment

  def getPickedObj(self, objects):
    if not objects:
      return None
    for obj in objects:
      # check the contact force normal to count the horizontal contact points
      contact_points = pb.getContactPoints(self.id, obj.object_id, self.finger_a_index) + pb.getContactPoints(self.id, obj.object_id, self.finger_b_index)
      horizontal = list(filter(lambda p: abs(p[7][2]) < 0.2, contact_points))
      if len(horizontal) >= 2:
        return obj

  def getGripperImg(self, img_size, workspace_size, obs_size_m):
    gripper_state = self.getGripperOpenRatio()
    gripper_rz = pb.getEulerFromQuaternion(self._getEndEffectorRotation())[2]

    im = np.zeros((img_size, img_size))
    gripper_half_size = 4 * workspace_size / obs_size_m
    gripper_half_size = math.ceil(gripper_half_size / 128 * img_size)
    gripper_max_open = 36 * workspace_size / obs_size_m

    anchor = (img_size // 2)
    d = int(gripper_max_open / 128 * img_size * gripper_state)
    im[int(anchor - d // 2 - gripper_half_size):int(anchor - d // 2 + gripper_half_size), int(anchor - gripper_half_size):int(anchor + gripper_half_size)] = 1
    im[int(anchor + d // 2 - gripper_half_size):int(anchor + d // 2 + gripper_half_size), int(anchor - gripper_half_size):int(anchor + gripper_half_size)] = 1
    im = rotate(im, np.rad2deg(gripper_rz), reshape=False, mode='nearest', order=0)

    return im

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot, self.ll, self.ul, self.jr, maxNumIterations=100, residualThreshold=1e-5)[:self.num_dofs]

  def _getGripperJointPosition(self):
    p1 = pb.getJointState(self.id, self.finger_a_index)[0]
    p2 = pb.getJointState(self.id, self.finger_b_index)[0]
    return p1, p2

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                 #targetVelocities=[0.]*num_motors,
                                 forces=self.max_forces,
                                 positionGains=[self.position_gain]*num_motors,
                                 #velocityGains=[1.0]*num_motors)
                                 )

  def _sendGripperCommand(self, target_pos1, target_pos2, force=10):
    pb.setJointMotorControlArray(self.id,
                                 [self.finger_a_index, self.finger_b_index],
                                 pb.POSITION_CONTROL,
                                 [target_pos1, target_pos2],
                                 forces=[force, force])
