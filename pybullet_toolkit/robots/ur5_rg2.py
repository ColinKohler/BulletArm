import os
import copy
import numpy as np
import numpy.random as npr
from collections import deque

import pybullet as pb
import pybullet_data

import helping_hands_rl_envs
import time

class UR5_RG2(object):
  '''

  '''
  def __init__(self):
    # Setup arm and gripper variables
    self.max_forces = [150, 150, 150, 28, 28, 28, 30, 30]
    self.gripper_close_force = [30] * 2
    self.gripper_open_force = [30] * 2
    self.end_effector_index = 12

    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0.]

    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)

  def reset(self):
    ''''''
    ur5_urdf_filepath = os.path.join(self.root_dir, 'urdf/ur5/ur5_w_simple_gripper.urdf')
    self.id = pb.loadURDF(ur5_urdf_filepath, [0,0,0], [0,0,0,1])
    self.is_holding = False
    self.gripper_closed = False
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]

    self.gripper_joint_names = list()
    self.gripper_joint_indices = list()
    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(1, 7):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)
      elif i in range(7, 9):
        self.gripper_joint_names.append(str(joint_info[1]))
        self.gripper_joint_indices.append(i)

  def saveState(self):
    self.state = {
      'is_holding': self.is_holding,
      'gripper_closed': self.gripper_closed
    }

  def restoreState(self):
    self.is_holding = self.state['is_holding']
    self.gripper_closed = self.state['gripper_closed']
    if self.gripper_closed:
      self._sendGripperCloseCommand()
    else:
      self._sendGripperOpenCommand()

  def pick(self, pos, rot, offset, dynamic=True):
    ''''''
    # Setup pre-grasp pos and default orientation
    self.openGripper()
    pre_pos = copy.copy(pos)
    pre_pos[2] += offset
    # rot = pb.getQuaternionFromEuler([np.pi/2.,-np.pi,np.pi/2])
    pre_rot = rot

    # Move to pre-grasp pose and then grasp pose
    self.moveTo(pre_pos, pre_rot, dynamic)
    self.moveTo(pos, rot, dynamic)

    # Grasp object and lift up to pre pose
    gripper_fully_closed = self.closeGripper()
    if gripper_fully_closed:
      self.moveTo(pre_pos, pre_rot, dynamic)
    else:
      self.moveTo(pre_pos, pre_rot, dynamic=True)
    if gripper_fully_closed: self.openGripper()

    self.is_holding = not gripper_fully_closed

  def place(self, pos, rot, offset, dynamic=True):
    ''''''
    # Setup pre-grasp pos and default orientation
    pre_pos = copy.copy(pos)
    pre_pos[2] += offset
    pre_rot = pb.getQuaternionFromEuler([0, np.pi, 0])

    # Move to pre-grasp pose and then grasp pose
    self.moveTo(pre_pos, pre_rot, True)
    self.moveTo(pos, rot, True)

    # Grasp object and lift up to pre pose
    self.openGripper()
    self.moveTo(pre_pos, pre_rot, dynamic)

    self.is_holding = False

  def moveTo(self, pos, rot, dynamic=True):
    ''''''
    close_enough = False
    outer_it = 0
    threshold = 1e-3
    max_outer_it = 100
    max_inner_it = 1000

    while not close_enough and outer_it < max_outer_it:
      ik_solve = pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)[:-2]
      if dynamic:
        self._sendPositionCommand(ik_solve)
        past_joint_pos = deque(maxlen=5)
        joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
        joint_pos = list(zip(*joint_state))[0]
        inner_it = 0
        while not np.allclose(joint_pos, ik_solve, atol=1e-2) and inner_it < max_inner_it:
          pb.stepSimulation()
          inner_it += 1
          # Check to see if the arm can't move any close to the desired joint position
          if len(past_joint_pos) == 5 and np.allclose(past_joint_pos[-1], past_joint_pos, atol=1e-3):
            break
          past_joint_pos.append(joint_pos)
          joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
          joint_pos = list(zip(*joint_state))[0]
      else:
        self._setJointPoses(ik_solve)

      ls = pb.getLinkState(self.id, self.end_effector_index)
      new_pos = list(ls[4])
      new_rot = list(ls[5])
      close_enough = np.allclose(np.array(new_pos + new_rot), np.array(list(pos) + list(rot)), atol=threshold)
      outer_it += 1

  def closeGripper(self):
    ''''''
    p1 = pb.getJointState(self.id, 10)[0]
    p2 = pb.getJointState(self.id, 11)[0]
    limit = 0.036
    self._sendGripperCloseCommand()
    self.gripper_closed = True
    it = 0
    while (limit-p1) + (limit-p2) > 0.001:
    # while p1 < 0.036:
      pb.stepSimulation()
      it += 1
      if it > 100:
        return False
      p1_ = pb.getJointState(self.id, 10)[0]
      p2_ = pb.getJointState(self.id, 11)[0]
      if p1 >= p1_ and p2 >= p2_:
        return False
      p1 = p1_
      p2 = p2_
    return True

  def checkGripperClosed(self):
    limit = 0.036
    p1 = pb.getJointState(self.id, 10)[0]
    p2 = pb.getJointState(self.id, 11)[0]
    if (limit - p1) + (limit - p2) > 0.001:
      self.is_holding = True
    else:
      self.is_holding = False

  def openGripper(self):
    ''''''
    p1 = pb.getJointState(self.id, 10)[0]
    self._sendGripperOpenCommand()
    self.gripper_closed = False
    it = 0
    while p1 > 0.0:
      pb.stepSimulation()
      it += 1
      if it > 100:
        return False
      p1 = pb.getJointState(self.id, 10)[0]

  def _getEndEffectorPosition(self):
    ''''''
    state = pb.getLinkState(self.id, self.end_effector_index)
    return np.array(state[4])

  def _getEndEffectorRotation(self):
    state = pb.getLinkState(self.id, self.end_effector_index)
    return np.array(state[5])

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
                                 [0.]*num_motors, self.max_forces[:-2], [0.01]*num_motors, [1.0]*num_motors)

  def _sendGripperCloseCommand(self):
    pb.setJointMotorControl2(self.id, 10, pb.POSITION_CONTROL, targetPosition=0.037, force=self.gripper_close_force[0])
    pb.setJointMotorControl2(self.id, 11, pb.POSITION_CONTROL, targetPosition=0.037, force=self.gripper_close_force[1])

  def _sendGripperOpenCommand(self):
    pb.setJointMotorControl2(self.id, 10, pb.POSITION_CONTROL, targetPosition=-0.01, force=self.gripper_close_force[0])
    pb.setJointMotorControl2(self.id, 11, pb.POSITION_CONTROL, targetPosition=-0.01, force=self.gripper_close_force[1])

  def _setJointPoses(self, q_poses):
    ''''''
    for i in range(len(q_poses)):
      motor = self.arm_joint_indices[i]
      pb.resetJointState(self.id, motor, q_poses[i])

    self._sendPositionCommand(q_poses)
