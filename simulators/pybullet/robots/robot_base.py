import os
import copy
import math
import numpy as np
import numpy.random as npr
from collections import deque
from abc import abstractmethod

import pybullet as pb
import pybullet_data

import helping_hands_rl_envs
import time

from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util
from helping_hands_rl_envs.simulators.pybullet.utils import object_generation
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class RobotBase:
  def __init__(self):
    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    self.id = None
    self.num_joints = None
    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    self.home_positions = None
    self.home_positions_joint = None
    self.end_effector_index = None
    self.holding_obj = None
    self.gripper_closed = False
    self.state = {
      'holding_obj': self.holding_obj,
      'gripper_closed': self.gripper_closed
    }

  def saveState(self):
    self.state = {
      'holding_obj': self.holding_obj,
      'gripper_closed': self.gripper_closed
    }

  def restoreState(self):
    self.holding_obj = self.state['holding_obj']
    self.gripper_closed = self.state['gripper_closed']
    if self.gripper_closed:
      self.closeGripper(max_it=0)
    else:
      self.openGripper()

  def getPickedObj(self, objects):
    if not objects:
      return None
    end_pos = self._getEndEffectorPosition()
    sorted_obj = sorted(objects, key=lambda o: np.linalg.norm(end_pos-o.getPosition()))
    obj_pos = sorted_obj[0].getPosition()
    if np.linalg.norm(end_pos[:-1]-obj_pos[:-1]) < 0.05 and np.abs(end_pos[-1]-obj_pos[-1]) < 0.025:
      return sorted_obj[0]

  def pick(self, pos, rot, offset, dynamic=True, objects=None, simulate_grasp=True):
    ''''''
    # Setup pre-grasp pos and default orientation
    self.openGripper()
    pre_pos = copy.copy(pos)
    pre_pos[2] += offset
    # rot = pb.getQuaternionFromEuler([np.pi/2.,-np.pi,np.pi/2])
    pre_rot = rot

    # Move to pre-grasp pose and then grasp pose
    self.moveTo(pre_pos, pre_rot, dynamic)
    if simulate_grasp:
      self.moveTo(pos, rot, True)
      # Grasp object and lift up to pre pose
      gripper_fully_closed = self.closeGripper()
      if gripper_fully_closed:
        self.openGripper()
        self.moveTo(pre_pos, pre_rot, dynamic)
      else:
        self.moveTo(pre_pos, pre_rot, True)
        self.adjustGripperCommand()
        for i in range(10):
          pb.stepSimulation()
        self.holding_obj = self.getPickedObj(objects)

    else:
      self.moveTo(pos, rot, dynamic)
      self.holding_obj = self.getPickedObj(objects)

    self.moveToJ(self.home_positions_joint, dynamic)
    self.checkGripperClosed()

  def place(self, pos, rot, offset, dynamic=True, simulate_grasp=True):
    ''''''
    # Setup pre-grasp pos and default orientation
    pre_pos = copy.copy(pos)
    pre_pos[2] += offset
    pre_rot = rot

    # Move to pre-grasp pose and then grasp pose
    self.moveTo(pre_pos, pre_rot, dynamic)
    if simulate_grasp:
      self.moveTo(pos, rot, True)
    else:
      self.moveTo(pos, rot, dynamic)

    # Grasp object and lift up to pre pose
    self.openGripper()
    self.holding_obj = None
    self.moveTo(pre_pos, pre_rot, dynamic)
    self.moveToJ(self.home_positions_joint, dynamic)

  def moveTo(self, pos, rot, dynamic=True):
    if dynamic or not self.holding_obj:
      self._moveToCartesianPose(pos, rot, dynamic)
    else:
      self._teleportArmWithObj(pos, rot)

  def moveToJ(self, pose, dynamic=True):
    if dynamic or not self.holding_obj:
      self._moveToJointPose(pose, dynamic)
    else:
      self._teleportArmWithObjJointPose(pose)

  @abstractmethod
  def openGripper(self):
    raise NotImplementedError

  @abstractmethod
  def closeGripper(self, max_it=100):
    raise NotImplementedError

  @abstractmethod
  def checkGripperClosed(self):
    raise NotImplementedError

  def _moveToJointPose(self, target_pose, dynamic=True, max_it=1000):
    if dynamic:
      self._sendPositionCommand(target_pose)
      past_joint_pos = deque(maxlen=5)
      joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
      joint_pos = list(zip(*joint_state))[0]
      n_it = 0
      while not np.allclose(joint_pos, target_pose, atol=1e-2) and n_it < max_it:
        pb.stepSimulation()
        n_it += 1
        # Check to see if the arm can't move any close to the desired joint position
        if len(past_joint_pos) == 5 and np.allclose(past_joint_pos[-1], past_joint_pos, atol=1e-3):
          break
        past_joint_pos.append(joint_pos)
        joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
        joint_pos = list(zip(*joint_state))[0]

    else:
      self._setJointPoses(target_pose)

  def _moveToCartesianPose(self, pos, rot, dynamic=True):
    close_enough = False
    outer_it = 0
    threshold = 1e-3
    max_outer_it = 10
    max_inner_it = 1000

    while not close_enough and outer_it < max_outer_it:
      ik_solve = self._calculateIK(pos, rot)
      self._moveToJointPose(ik_solve, dynamic, max_inner_it)

      ls = pb.getLinkState(self.id, self.end_effector_index)
      new_pos = list(ls[4])
      new_rot = list(ls[5])
      close_enough = np.allclose(np.array(new_pos + new_rot), np.array(list(pos) + list(rot)), atol=threshold)
      outer_it += 1

  @abstractmethod
  def _calculateIK(self, pos, rot):
    raise NotImplementedError

  def _teleportArmWithObj(self, pos, rot):
    if not self.holding_obj:
      self._moveToCartesianPose(pos, rot, False)
      return

    end_pos = self._getEndEffectorPosition()
    end_rot = self._getEndEffectorRotation()
    obj_pos, obj_rot = self.holding_obj.getPose()
    oTend = pybullet_util.getMatrix(end_pos, end_rot)
    oTobj = pybullet_util.getMatrix(obj_pos, obj_rot)
    endTobj = np.linalg.inv(oTend).dot(oTobj)

    self._moveToCartesianPose(pos, rot, False)
    end_pos_ = self._getEndEffectorPosition()
    end_rot_ = self._getEndEffectorRotation()
    oTend_ = pybullet_util.getMatrix(end_pos_, end_rot_)
    oTobj_ = oTend_.dot(endTobj)
    obj_pos_ = oTobj_[:3, -1]
    obj_rot_ = transformations.quaternion_from_matrix(oTobj_)

    self.holding_obj.resetPose(obj_pos_, obj_rot_)

  def _teleportArmWithObjJointPose(self, joint_pose):
    if not self.holding_obj:
      self._moveToJointPose(joint_pose, False)
      return

    end_pos = self._getEndEffectorPosition()
    end_rot = self._getEndEffectorRotation()
    obj_pos, obj_rot = self.holding_obj.getPose()
    oTend = pybullet_util.getMatrix(end_pos, end_rot)
    oTobj = pybullet_util.getMatrix(obj_pos, obj_rot)
    endTobj = np.linalg.inv(oTend).dot(oTobj)

    self._moveToJointPose(joint_pose, False)
    end_pos_ = self._getEndEffectorPosition()
    end_rot_ = self._getEndEffectorRotation()
    oTend_ = pybullet_util.getMatrix(end_pos_, end_rot_)
    oTobj_ = oTend_.dot(endTobj)
    obj_pos_ = oTobj_[:3, -1]
    obj_rot_ = transformations.quaternion_from_matrix(oTobj_)

    self.holding_obj.resetPose(obj_pos_, obj_rot_)

  def _getEndEffectorPosition(self):
    ''''''
    state = pb.getLinkState(self.id, self.end_effector_index)
    return np.array(state[4])

  def _getEndEffectorRotation(self):
    state = pb.getLinkState(self.id, self.end_effector_index)
    return np.array(state[5])

  @abstractmethod
  def _getGripperJointPosition(self):
    raise NotImplementedError

  @abstractmethod
  def _sendPositionCommand(self, commands):
    raise NotImplementedError

  @abstractmethod
  def adjustGripperCommand(self):
    raise NotImplementedError

  def _setJointPoses(self, q_poses):
    ''''''
    for i in range(len(q_poses)):
      motor = self.arm_joint_indices[i]
      pb.resetJointState(self.id, motor, q_poses[i])

    self._sendPositionCommand(q_poses)
