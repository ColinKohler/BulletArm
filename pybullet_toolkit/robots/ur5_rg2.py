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
    self.max_forces = [150, 150, 150, 28, 28, 28, 100, 100]
    self.gripper_close_force = [100] * 2
    self.gripper_open_force = [100] * 2
    self.end_effector_index = 12

    self.home_positions = [0., 0., -2.137, 1.432, -0.915, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0.]

    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)

  def reset(self):
    ''''''
    ur5_urdf_filepath = os.path.join(self.root_dir, 'urdf/ur5/ur5_w_simple_gripper.urdf')
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

  def pick(self, pos, rot, offset, dynamic=True):
    ''''''
    # Setup pre-grasp pos and default orientation
    pre_pos = copy.copy(pos)
    pre_pos[2] += offset
    # rot = pb.getQuaternionFromEuler([np.pi/2.,-np.pi,np.pi/2])
    pre_rot = rot

    # Move to pre-grasp pose and then grasp pose
    self.moveTo(pre_pos, pre_rot, dynamic)
    self.moveTo(pos, rot, dynamic)

    # Grasp object and lift up to pre pose
    gripper_fully_closed = self.closeGripper()
    self.moveTo(pre_pos, pre_rot, dynamic)
    if gripper_fully_closed: self.openGripper()

    self.is_holding = not gripper_fully_closed

  def place(self, pos, rot, offset, dynamic=True):
    ''''''
    # Setup pre-grasp pos and default orientation
    pre_pos = copy.copy(pos)
    pre_pos[2] += offset
    pre_rot = pb.getQuaternionFromEuler([0, np.pi, 0])

    # Move to pre-grasp pose and then grasp pose
    self.moveTo(pre_pos, pre_rot, dynamic)
    self.moveTo(pos, rot, dynamic)

    # Grasp object and lift up to pre pose
    self.openGripper()
    self.moveTo(pre_pos, pre_rot, dynamic)

    self.is_holding = False

  def moveTo(self, pos, rot, dynamic=True):
    ''''''
    closeEnough = False
    it = 0
    threshold = 1e-3
    max_iteration = 100

    while not closeEnough and it < max_iteration:
      ik_solve = pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)
      if dynamic:
        self._sendPositionCommand(ik_solve)
        past_joint_pos = deque(maxlen=5)
        joint_state = pb.getJointStates(self.id, self.motor_indices)
        joint_pos = list(zip(*joint_state))[0]
        while not np.allclose(joint_pos, ik_solve, atol=1e-2):
          pb.stepSimulation()
          # Check to see if the arm can't move any close to the desired joint position
          if len(past_joint_pos) == 5 and np.allclose(past_joint_pos[-1], past_joint_pos, atol=1e-3):
            break
          past_joint_pos.append(joint_pos)
          joint_state = pb.getJointStates(self.id, self.motor_indices)
          joint_pos = list(zip(*joint_state))[0]
      else:
        self._setJointPoses(ik_solve)

      ls = pb.getLinkState(self.id, self.end_effector_index)
      new_pos = list(ls[4])
      new_rot = list(ls[5])
      diff = np.array(new_pos + new_rot) - np.array(list(pos) + list(rot))
      diff = np.abs(diff).mean()
      closeEnough = (diff < threshold)
      it += 1
    pass

    # # ik_solve = pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot)
    # ik_solve = self.accurateCalculateInverseKinematics(pos, rot, 1e-4, 1000)
    # if dynamic:
    #   ee_pos = self._getEndEffectorPosition()
    #   ee_rot = self._getEndEffectorRotation()
    #   self._sendPositionCommand(ik_solve)
    #   past_ee_pos = deque(maxlen=5)
    #   past_ee_rot = deque(maxlen=5)
    #   while not (np.allclose(ee_pos, pos, atol=0.01) and np.allclose(ee_rot, rot, atol=0.01)):
    #     # time.sleep(0.005)
    #     pb.stepSimulation()
    #
    #     # Check to see if the arm can't move any close to the desired position
    #     if len(past_ee_pos) == 5 and np.allclose(past_ee_pos[0], past_ee_pos, 1e-3) \
    #         and len(past_ee_rot) == 5 and np.allclose(past_ee_rot[0], past_ee_rot, 1e-3):
    #     # if len(past_ee_pos) == 5 and np.allclose(past_ee_pos[0], past_ee_pos):
    #       js = pb.getJointStates(self.id, self.motor_indices)
    #       jp = list(zip(*js))[0]
    #       error = np.array(ik_solve) - jp
    #       print(max(error))
    #       break
    #
    #     past_ee_pos.append(ee_pos)
    #     past_ee_rot.append(ee_rot)
    #     ee_pos = self._getEndEffectorPosition()
    #     ee_rot = self._getEndEffectorRotation()
    # else:
    #   self._setJointPoses(ik_solve)

  def closeGripper(self):
    ''''''
    p1 = pb.getJointState(self.id, 10)[0]
    pb.setJointMotorControlArray(self.id, [10,11], pb.VELOCITY_CONTROL, targetVelocities=[1.0, 1.0], forces=self.gripper_close_force)
    while p1 < 0.036:
      pb.stepSimulation()
      p1_ = pb.getJointState(self.id, 10)[0]
      if p1 >= p1_:
        return False
      p1 = p1_

    return True

  def openGripper(self):
    ''''''
    p1 = pb.getJointState(self.id, 10)[0]
    pb.setJointMotorControlArray(self.id, [10,11], pb.VELOCITY_CONTROL, targetVelocities=[-1.0, -1.0], forces=self.gripper_open_force)

    while p1 > 0.0:
      pb.stepSimulation()
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
    num_motors = len(self.motor_indices)
    pb.setJointMotorControlArray(self.id, self.motor_indices, pb.POSITION_CONTROL, commands,
                                 [0.]*num_motors, self.max_forces, [0.01]*num_motors, [1.0]*num_motors)

  def _setJointPoses(self, q_poses):
    ''''''
    for i in range(len(q_poses)):
      motor = self.motor_indices[i]
      pb.resetJointState(self.id, motor, q_poses[i])
