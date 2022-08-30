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
    elif gripper == 'hydrostatic':
      self.gripper = Hydrostatic()
      self.urdf_filepath = constants.UR5_HYDROSTATIC_PATH
    elif gripper == 'openhand_vf':
      self.gripper = OpenHandVF()
      self.urdf_filepath = constants.UR5_OPENHAND_VF_PATH
    self.end_effector_index = self.gripper.end_effector_index

    # Setup arm
    self.num_dofs = 6
    self.wrist_index = 5
    self.max_forces = [150, 150, 150, 28, 28, 28]
    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0.]

  def initialize(self):
    ''''''
    self.id = pb.loadURDF(self.urdf_filepath, [0,0,0.1], [0,0,0,1], useFixedBase=True)
    pb.resetBasePositionAndOrientation(self.id, [-0.1,0,0], [0,0,0,1])

    self.gripper.closed = False
    self.pre_holding_obj = None

    # Enable force sensors
    pb.enableJointForceTorqueSensor(self.id, self.wrist_index)
    self.gripper.enableFingerForceTorqueSensors(self.id)

    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(1, self.num_dofs+1):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)

    # Zero force out
    self.force_history = list()
    pb.stepSimulation()
    force, moment = self.getWristForce()
    self.zero_force = np.concatenate((force, moment))

    self.openGripper()

  def reset(self):
    self.gripper.closed = False
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
    self.gripper.getFingerForce(self.id)

  def controlGripper(self, open_ratio, max_it=100):
    self.gripper.contolGripper(open_ratio, max_it=max_it)

  def openGripper(self):
    return self.gripper.openGripper()

  def closeGripper(self):
    return self.gripper.closeGripper()

  def getGripperImg(self, img_size, workspace_size, obs_size_m):
    return self.gripper.getGripperImg(img_size, workspace_size, obs_size_m)

  def adjustGripperCommand(self):
    self.gripper.adjustGripperCommand()

  def checkGripperClosed(self):
    return self.gripper.checkGripperClosed()

  def gripperHasForce(self):
    return self.gripper.hasForce()

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(
      self.id,
      self.end_effector_index,
      pos,
      targetOrientation=rot)[:self.num_dofs]

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(
      self.id,
      self.arm_joint_indices,
      pb.POSITION_CONTROL,
      commands,
      targetVelocities=[0.] * num_motors,
      forces=self.max_forces,
      positionGains=[self.position_gain] * num_motors,
      velocityGains=[1.0] * num_motors)
