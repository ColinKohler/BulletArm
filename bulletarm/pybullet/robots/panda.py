import os
import numpy as np
import pybullet as pb

from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.robots.robot_base import RobotBase
from bulletarm.pybullet.robots.gripper_base import GripperBase

class Panda(RobotBase):
  '''

  '''
  def __init__(self):
    super().__init__()
    self.home_positions = [-0.60, -0.14, 0.59, -2.40, 0.11, 2.28, -1, 0.0, 0, 0, 0, 0, 0, 0, 0]
    self.home_positions_joint = self.home_positions[:7]
    self.position_gain = 1.0

    self.num_dofs = 7
    self.wrist_index = 8
    self.finger_idxs = [10, 12]
    self.end_effector_index = 13
    self.gripper_z_offset = 0.08
    self.gripper_joint_limit = [0, 0.04]
    self.gripper = GripperBase(self.finger_idxs, self.gripper_z_offset, self.gripper_joint_limit)

    self.max_torque = [25., 25., 25., 25., 25., 25., 25.]

    self.ll = [-2.9671, -1.8326, -2.9671, -3.1416, -2.9671, -0.0873, -2.9671]
    self.ul = [ 2.9671,  1.8326,  2.9671,  0,       2.9671,  3.8223,  2.9671]
    self.ml = [0, 0, 0, -1.5708, 0, 1.8675, 0]
    self.jr = [5.9342, 3.6652, 5.9342, 3.141, 5.9342, 3.9096, 5.9342]

    self.urdf_filepath = os.path.join(constants.URDF_PATH, 'franka_panda/panda.urdf')

  def initialize(self):
    ''''''
    self.id = pb.loadURDF(self.urdf_filepath, useFixedBase=True)
    super().initialize()
    c = pb.createConstraint(self.id,
                            self.finger_idxs[0],
                            self.id,
                            self.finger_idxs[1],
                            jointType=pb.JOINT_GEAR,
                            jointAxis=[1, 0, 0],
                            parentFramePosition=[0, 0, 0],
                            childFramePosition=[0, 0, 0])
    pb.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50)
