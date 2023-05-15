import os
import pybullet as pb
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.robots.robot_base import RobotBase
from bulletarm.pybullet.robots.gripper_base import GripperBase

class UR5_Simple(RobotBase):
  '''

  '''
  def __init__(self):
    super(UR5_Simple, self).__init__()
    # Setup arm and gripper variables
    self.max_forces = [150, 150, 150, 28, 28, 28, 30, 30]
    self.gripper_close_force = [30] * 2
    self.end_effector_index = 12

    self.wrist_index = 7
    self.num_dofs = 7
    self.finger_idxs = [10, 11]
    self.adjust_gripper_offset = -0.01
    
    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0.]
    self.home_positions_joint = self.home_positions[1:7]

    self.gripper_joint_limit = [0.036, 0.0]

  def _calculateIK(self, pos, rot):
    return pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:-2]

  def initialize(self):
    ur5_urdf_filepath = os.path.join(constants.URDF_PATH, 'ur5/ur5_simple_gripper.urdf')
    self.id = pb.loadURDF(ur5_urdf_filepath, [0,0,0], [0,0,0,1])
    self.gripper_closed = False
    self.holding_obj = None
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    self.gripper = Ur5Gripper(self.finger_idxs, 0.0, self.gripper_joint_limit, self.adjust_gripper_offset)
    super().initialize()

  def _sendPositionCommand(self, commands):
    ''''''
    self.arm_joint_indices = [1, 2, 3, 4, 5, 6]
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(
      bodyIndex=self.id,
      jointIndices=self.arm_joint_indices,
      controlMode=pb.POSITION_CONTROL,
      targetPositions=commands,
      forces=self.max_forces[:-2],
      positionGains=[self.position_gain]*num_motors,
    )

class Ur5Gripper(GripperBase):
  '''

  '''
  def __init__(self, finger_idxs, z_offset, joint_limit, adjust_offset):
    super().__init__(finger_idxs, z_offset, joint_limit)
    self.adjust_offset = adjust_offset

  def adjustCommand(self):
    p1, p2 = self._getJointPosition()
    mean = (p1 + p2) / 2 - self.adjust_offset
    self._sendCommand(mean, mean)

  def _sendCommand(self, target_pos_1, target_pos_2, force=10):
    pb.setJointMotorControlArray(
      self.robot_id,
      [self.finger_idxs[0], self.finger_idxs[1]],
      pb.POSITION_CONTROL,
      [target_pos_1, target_pos_2],
      forces=[30, 30],
      positionGains=[0.02]*2,
      velocityGains=[1.0]*2
    )

  def open(self, max_it=100, force=10):
    ''''''
    self.closed = False
    self.holding_obj = None
    return self.control(0, max_it=max_it, force=force)

  def close(self, max_it=100, force=10):
    ''''''
    self.closed = True
    return self.control(1, max_it=max_it, force=force)