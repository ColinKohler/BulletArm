import os
import copy
import time
import numpy as np
import numpy.random as npr
from collections import deque

import pybullet as pb
import pybullet_data

import helping_hands_rl_envs

class UR5_RG2(object):
  '''

  '''
  def __init__(self):
    # Setup arm and gripper variables
    self.max_forces = [150, 150, 150, 28, 28, 28, 10, 10, 10, 10, 10, 10]
    self.gripper_close_force = [25] * 6
    self.gripper_open_force = [25] * 6

    self.end_effector_index = 9
    self.gripper_index = 19

    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0.,
                           0, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)

  def reset(self):
    ''''''
    ur5_urdf_filepath = os.path.join(self.root_dir, 'urdf/ur5/ur5_w_robotiq_85_gripper.urdf')
    self.id = pb.loadURDF(ur5_urdf_filepath, [0,0,0], [0,0,0,1])
    self.is_holding = False
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    self.motor_names = list()
    self.motor_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      q_index = joint_info[3]
      if q_index > -1:
        self.motor_names.append(str(joint_info[1]))
        self.motor_indices.append(i)

  def pick(self, pos, offset, dynamic=True):
    ''''''
    # Setup pre-grasp pos and default orientation
    pre_pos = copy.copy(pos)
    pre_pos[2] += offset
    # rot = pb.getQuaternionFromEuler([np.pi/2.,-np.pi,np.pi/2])
    rot = pb.getQuaternionFromEuler([0,np.pi,0])

    # Move to pre-grasp pose and then grasp pose
    # time.sleep(1)
    self.moveTo(pre_pos, rot, dynamic)
    # time.sleep(1)
    self.moveTo(pos, rot, dynamic)

    # Grasp object and lift up to pre pose
    # time.sleep(1)
    gripper_fully_closed = self.closeGripper()
    # time.sleep(1)
    if gripper_fully_closed: self.openGripper()
    # time.sleep(1)
    self.moveTo(pre_pos, rot, dynamic)

    self.is_holding = not gripper_fully_closed

  def place(self, pos, offset, dynamic=True):
    ''''''
    # Setup pre-grasp pos and default orientation
    pre_pos = copy.copy(pos)
    pre_pos[2] += offset
    rot = pb.getQuaternionFromEuler([np.pi/2.,-np.pi,np.pi/2])

    # Move to pre-grasp pose and then grasp pose
    self.moveTo(pre_pos, rot, dynamic)
    self.moveTo(pos, rot, dynamic)

    # Grasp object and lift up to pre pose
    self.openGripper()
    self.moveTo(pre_pos, rot, dynamic)

    self.is_holding = False

  def moveTo(self, pos, rot, dynamic=True):
    ''''''
    ik_solve = pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)


    if dynamic:
      ee_pos = self._getEndEffectorPosition()
      self._sendPositionCommand(ik_solve)
      past_ee_pos = deque(maxlen=5)
      while not np.allclose(ee_pos, pos, atol=0.01):
        pb.stepSimulation()

        # Check to see if the arm can't move any close to the desired position
        if len(past_ee_pos) == 5 and np.allclose(past_ee_pos[0], past_ee_pos):
          break

        past_ee_pos.append(ee_pos)
        ee_pos = self._getEndEffectorPosition()
    else:
      self._setJointPoses(ik_solve)

  def closeGripper(self):
    ''''''
    p1 = pb.getJointState(self.id, 10)[0]
    pb.setJointMotorControlArray(self.id, [10,12,14,15,17,19], pb.VELOCITY_CONTROL, targetVelocities=[1.0]*6, forces=self.gripper_close_force)
    while p1 < 0.4:
      pb.stepSimulation()
      p1_ = pb.getJointState(self.id, 10)[0]
      if p1 >= p1_:
        return False
      p1 = p1_

    return True

  def openGripper(self):
    ''''''
    p1 = pb.getJointState(self.id, 10)[0]
    pb.setJointMotorControlArray(self.id, [10,12,14,15,17,19], pb.VELOCITY_CONTROL, targetVelocities=[-1.0]*6, forces=self.gripper_open_force)

    while p1 > 0.0:
      pb.stepSimulation()
      p1 = pb.getJointState(self.id, 10)[0]

  def _getEndEffectorPosition(self):
    ''''''
    state = pb.getLinkState(self.id, self.end_effector_index)
    return np.array(state[4])

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.motor_indices)
    pb.setJointMotorControlArray(self.id, self.motor_indices, pb.POSITION_CONTROL, commands,
                                 [0.]*num_motors, self.max_forces, [0.01]*num_motors, [1]*num_motors)

  def _setJointPoses(self, q_poses):
    ''''''
    for i in range(len(q_poses)):
      motor = self.motor_indices[i]
      pb.resetJointState(self.id, motor, q_poses[i])
