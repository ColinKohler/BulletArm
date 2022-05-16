'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import os
import copy
import math
import numpy as np
import numpy.random as npr
from collections import deque
from abc import abstractmethod

import pybullet as pb
import pybullet_data

import bulletarm
import time

from bulletarm.pybullet.utils import pybullet_util
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import object_generation
from bulletarm.pybullet.utils import transformations

class RobotBase:
  '''
  Base Robot Class.
  '''
  def __init__(self):
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

    self.position_gain = 0.02
    self.adjust_gripper_after_lift = False

  def saveState(self):
    '''
    Set the robot state. The state consists of the object that is being held and
    if the gripper is closed.
    '''
    self.state = {
      'holding_obj': self.holding_obj,
      'gripper_closed': self.gripper_closed
    }

  def restoreState(self):
    '''
    Restores the robot to the previously saved state.
    '''
    self.holding_obj = self.state['holding_obj']
    self.gripper_closed = self.state['gripper_closed']
    if self.gripper_closed:
      self.closeGripper(max_it=0)
    else:
      self.openGripper()

  def getPickedObj(self, objects):
    '''
    Get the object which is currently being held by the gripper.

    Args:
      objects (numpy.array): Objects to check if are being held.

    Returns:
      (pybullet.objects.PybulletObject): Object being held.
    '''
    if not objects:
      return None

    for obj in objects:
      # check the contact force normal to count the horizontal contact points
      contact_points = pb.getContactPoints(self.id, obj.object_id)
      horizontal = list(filter(lambda p: abs(p[7][2]) < 0.3, contact_points))
      if len(horizontal) >= 2:
        return obj

    return None

  def pick(self, pos, rot, offset, dynamic=True, objects=None, simulate_grasp=True, top_down_approach=False):
    '''
    Execute a pick action at the given pose.

    Args:
      pos (numpy.array): Desired end effector position.
      rot (numpy.array): Desired end effector orientation.
      offset (float): Grasp offset along the approach axis of the grasp.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
      objects (numpy.array): List of objects which can be picked up. Defaults to None.
      simulate_grasp (bool): Simulate the grasping action. Defaults to True. If set to False, object
        will be grasped if it is within a small distance from the gripper.
      top_down_approach (bool): Force a top-down grasp action. Defaults to False. If set to True,
        approach vector will be set to top-down.
    '''
    self.openGripper()

    # Setup pre-grasp pose
    pre_pos = copy.copy(pos)
    pre_rot = copy.copy(rot)
    if top_down_approach:
      pre_pos[2] += offset
    else:
      m = np.array(pb.getMatrixFromQuaternion(rot)).reshape(3, 3)
      pre_pos += m[:, 2] * offset

    # Move to pre-grasp pose and then grasp pose
    self.moveTo(pre_pos, pre_rot, dynamic)
    if simulate_grasp:
      self.moveTo(pos, rot, True, pos_th=1e-3, rot_th=1e-3)

      # Close gripper, if fully closed (nothing grasped), open gripper
      gripper_fully_closed = self.closeGripper()
      if gripper_fully_closed:
        self.openGripper()

      # Adjust gripper command after moving to pre-grasp pose. Useful in cluttered domains.
      # This will increase grasp chance but gripper will shift while lifting object.
      if self.adjust_gripper_after_lift:
        self.moveTo(pre_pos, pre_rot, True)
        self.adjustGripperCommand()
      # Adjust gripper command before moving to pre-grasp pose.
      # This will increase gripper stabilization but will reduce grasp chance.
      else:
        self.adjustGripperCommand()
        self.moveTo(pre_pos, pre_rot, True)

      for i in range(100):
        pb.stepSimulation()
    else:
      self.moveTo(pos, rot, dynamic)

    self.holding_obj = self.getPickedObj(objects)
    self.moveToJ(self.home_positions_joint, dynamic)
    self.checkGripperClosed()

  def place(self, pos, rot, offset, dynamic=True, simulate_place=True, top_down_approach=False):
    '''
    Execute a place action at the given pose.

    Args:
      pos (numpy.array): Desired end effector position.
      rot (numpy.array): Desired end effector orientation.
      offset (float): Place offset along the approach axis of the grasp.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
      simulate_place (bool): Simulate the placing action. Defaults to True. If set to False, object
        will be grasped if it is within a small distance from the gripper.
      top_down_approach (bool): Force a top-down grasp action. Defaults to False. If set to True,
        approach vector will be set to top-down.
    '''
    # Setup pre-place pose
    pre_pos = copy.copy(pos)
    pre_rot = copy.copy(rot)
    if top_down_approach:
      pre_pos[2] += offset
    else:
      m = np.array(pb.getMatrixFromQuaternion(rot)).reshape(3, 3)
      pre_pos += m[:, 2] * offset

    # Move to pre-place pose and then place pose
    self.moveTo(pre_pos, pre_rot, dynamic)
    if simulate_place:
      self.moveTo(pos, rot, True, pos_th=1e-3, rot_th=1e-3)
    else:
      self.moveTo(pos, rot, dynamic, pos_th=1e-3, rot_th=1e-3)

    # Place object and lift up to pre pose
    self.openGripper()
    self.holding_obj = None
    self.moveTo(pre_pos, pre_rot, dynamic)
    self.moveToJ(self.home_positions_joint, dynamic)

  def push(self, pos, rot, offset, dynamic=True):
    '''
    Execute a push action at the given pose.

    Args:
      pos (numpy.array): Desired end effector position.
      rot (numpy.array): Desired end effector orientation.
      offset (float): Push offset along the approach axis of the push.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
    '''
    goal_pos = copy.copy(pos)
    m = np.array(pb.getMatrixFromQuaternion(rot)).reshape(3, 3)
    goal_pos += m[:, 1] * offset

    pre_pos = copy.copy(pos)
    m = np.array(pb.getMatrixFromQuaternion(rot)).reshape(3, 3)
    pre_pos -= m[:, 1] * 0.1

    self.closeGripper(primative=constants.PULL_PRIMATIVE)
    self.moveTo(pre_pos, rot, dynamic)
    self.moveTo(pos, rot, True)
    self.moveTo(goal_pos, rot, True)
    self.openGripper()
    self.moveToJ(self.home_positions_joint, dynamic)

  def pull(self, pos, rot, offset, dynamic=True):
    '''
    Execute a pull action at the given pose.

    Args:
      pos (numpy.array): Desired end effector position.
      rot (numpy.array): Desired end effector orientation.
      offset (float): Pull offset along the approach axis of the pull.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
    '''

    pre_pos = copy.copy(pos)
    m = np.array(pb.getMatrixFromQuaternion(rot)).reshape(3, 3)
    pre_pos += m[:, 2] * offset
    self.moveTo(pre_pos, rot, dynamic)
    # for mid in np.arange(0, offset, 0.05)[1:]:
    #   self.moveTo(pre_pos - m[:, 2] * mid, rot, True)
    self.moveTo(pos, rot, True)
    self.closeGripper(primative=constants.PULL_PRIMATIVE)
    # for mid in np.arange(0, offset, 0.05)[1:]:
    #   self.moveTo(pos + m[:, 2] * mid, rot, True)
    self.moveTo(pre_pos, rot, True)
    self.openGripper()
    self.moveToJ(self.home_positions_joint, dynamic)

  def roundPull(self, pos, rot, offset, radius, left=True, dynamic=True):
    '''
    Execute a pull action at the given pose. In a circular motion.

    Args:
      pos (numpy.array): Desired end effector position.
      rot (numpy.array): Desired end effector orientation.
      offset (float): Pull offset along the approach axis of the pull.
      radius (float): Radius for the circle in the circular pulling motion.
      left (bool): Rotate round pull action from the left. Defaults to True.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
    '''
    pre_pos = copy.copy(pos)
    m = np.eye(4)
    m[:3, :3] = np.array(pb.getMatrixFromQuaternion(rot)).reshape(3, 3)
    pre_pos += m[:3, 2] * offset
    waypoint_theta = np.linspace(0, np.pi/2, 10)
    waypoint_pos = []
    waypoint_rot = []
    for theta in waypoint_theta:
      dx = -np.sin(theta) * radius
      if left:
        dy = (1 - np.cos(theta)) * radius
      else:
        dy = -(1 - np.cos(theta)) * radius
      waypoint_pos.append((pos[0] + dx, pos[1] + dy, pos[2]))
      if left:
        m_prime = m.dot(transformations.euler_matrix(0, theta, 0))
        # m_prime = m.dot(transformations.euler_matrix(0, 0, 0))
      else:
        m_prime = m.dot(transformations.euler_matrix(0, -theta, 0))
      waypoint_rot.append(transformations.quaternion_from_matrix(m_prime))

    self.moveTo(pre_pos, rot, dynamic)
    self.moveTo(pos, rot, True)
    self.closeGripper(primative=constants.PULL_PRIMATIVE)
    for i in range(len(waypoint_theta)):
      self.moveTo(waypoint_pos[i], waypoint_rot[i], True)

    self.openGripper()
    self.moveToJ(self.home_positions_joint, dynamic)

  def moveTo(self, pos, rot, dynamic=True, pos_th=1e-3, rot_th=1e-3):
    '''
    Move the end-effector to the given pose.

    Args:
      pos (numpy.array): Desired end-effector position.
      rot (numpy.array): Desired end-effector orientation.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
      pos_th (float): Positional threshold for ending the movement. Defaults to 1e-3.
      rot_th (float): Rotational threshold for ending the movement. Defaults to 1e-3.
    '''
    if dynamic or not self.holding_obj:
      self._moveToCartesianPose(pos, rot, dynamic, pos_th, rot_th)
    else:
      self._teleportArmWithObj(pos, rot)

  def moveToJ(self, joint_pose, dynamic=True):
    '''
    Move the desired joint positions

    Args:
      joint_pose (numpy.array): Joint positions for each joint in the manipulator.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
    '''
    if dynamic or not self.holding_obj:
      self._moveToJointPose(joint_pose, dynamic)
    else:
      self._teleportArmWithObjJointPose(joint_pose)

  def _moveToJointPose(self, target_pose, dynamic=True, max_it=1000):
    '''
    Move the desired joint positions

    Args:
      joint_pose (numpy.array): Joint positions for each joint in the manipulator.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
      max_it (int): Maximum number of iterations the movement can take. Defaults to 10000.
    '''
    if dynamic:
      self._sendPositionCommand(target_pose)
      past_joint_pos = deque(maxlen=5)
      joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
      joint_pos = list(zip(*joint_state))[0]
      n_it = 0
      while not np.allclose(joint_pos, target_pose, atol=1e-3) and n_it < max_it:
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

  def _moveToCartesianPose(self, pos, rot, dynamic=True, pos_th=1e-3, rot_th=1e-3):
    '''
    Move the end effector to the desired cartesian pose.

    Args:
      pos (numpy.array): Desired end-effector position.
      rot (numpy.array): Desired end-effector orientation.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
      pos_th (float): Positional threshold for ending the movement. Defaults to 1e-3.
      rot_th (float): Rotational threshold for ending the movement. Defaults to 1e-3.
    '''

    close_enough = False
    outer_it = 0
    max_outer_it = 10
    max_inner_it = 100

    while not close_enough and outer_it < max_outer_it:
      ik_solve = self._calculateIK(pos, rot)
      self._moveToJointPose(ik_solve, dynamic, max_inner_it)

      ls = pb.getLinkState(self.id, self.end_effector_index)
      new_pos = list(ls[4])
      new_rot = list(ls[5])
      close_enough = np.allclose(np.array(new_pos), pos, atol=pos_th) and \
                     np.allclose(np.array(new_rot), rot, atol=rot_th)
      outer_it += 1

  def _teleportArmWithObj(self, pos, rot):
    '''
    Teleport the arm to the given pose along with the object that is being grasped.

    Args:
      pos (numpy.array): Desired end-effector position.
      rot (numpy.array): Desired end-effector orientation.
    '''
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

  def getEndToHoldingObj(self):
    '''
    Get the transfomration from the end effector to the object currently beinging held.

    Returns:
      numpy.array: Transformation matrix
    '''
    if not self.holding_obj:
      return np.zeros((4, 4))

    end_pos = self._getEndEffectorPosition()
    end_rot = self._getEndEffectorRotation()
    obj_pos, obj_rot = self.holding_obj.getPose()
    oTend = pybullet_util.getMatrix(end_pos, end_rot)
    oTobj = pybullet_util.getMatrix(obj_pos, obj_rot)
    endTobj = np.linalg.inv(oTend).dot(oTobj)

    return endTobj

  def _teleportArmWithObjJointPose(self, joint_pose):
    '''
    Directly set the arm to the given joint positions. Moves the object being held.

    Args:
      joint_pose (numpy.array): The joint positions for the manipulators
    '''
    if not self.holding_obj:
      self._moveToJointPose(joint_pose, False)
      return

    endTobj = self.getEndToHoldingObj()

    self._moveToJointPose(joint_pose, False)

    end_pos_ = self._getEndEffectorPosition()
    end_rot_ = self._getEndEffectorRotation()
    oTend_ = pybullet_util.getMatrix(end_pos_, end_rot_)
    oTobj_ = oTend_.dot(endTobj)
    obj_pos_ = oTobj_[:3, -1]
    obj_rot_ = transformations.quaternion_from_matrix(oTobj_)

    self.holding_obj.resetPose(obj_pos_, obj_rot_)

  def _getEndEffectorPosition(self):
    '''
    Get the current end effector position.

    Returns:
      (numpy.array): The end effector position.
    '''
    state = pb.getLinkState(self.id, self.end_effector_index)
    return np.array(state[4])

  def _getEndEffectorRotation(self):
    '''
    Get the current end effector orientation.

    Returns:
      (numpy.array): The end effector orientation.
    '''
    state = pb.getLinkState(self.id, self.end_effector_index)
    return np.array(state[5])

  def _setJointPoses(self, q_poses):
    '''
    Set the joints to the given positions.

    Args:
      q_poses (numpy.array): The joint positions.
    '''
    for i in range(len(q_poses)):
      motor = self.arm_joint_indices[i]
      pb.resetJointState(self.id, motor, q_poses[i])

    self._sendPositionCommand(q_poses)

  def plotEndEffectorFrame(self):
    '''
    Plot the end effector's frame in the PyBullet GUI.

    Returns:
      () :
    '''
    line_id1 = pb.addUserDebugLine(self._getEndEffectorPosition(),
                                  self._getEndEffectorPosition() + 0.1 * transformations.quaternion_matrix(
                                    self._getEndEffectorRotation())[:3, 0], (1, 0, 0))
    line_id2 = pb.addUserDebugLine(self._getEndEffectorPosition(),
                                  self._getEndEffectorPosition() + 0.1 * transformations.quaternion_matrix(
                                    self._getEndEffectorRotation())[:3, 1], (0, 1, 0))
    line_id3 = pb.addUserDebugLine(self._getEndEffectorPosition(),
                                  self._getEndEffectorPosition() + 0.1 * transformations.quaternion_matrix(
                                    self._getEndEffectorRotation())[:3, 2], (0, 0, 1))
    return line_id1, line_id2, line_id3

  #===========================================================================#
  #           Abstract functions sub-class robots must implement              #
  #===========================================================================#
  @abstractmethod
  def _calculateIK(self, pos, rot):
    raise NotImplementedError

  @abstractmethod
  def openGripper(self):
    raise NotImplementedError

  @abstractmethod
  def closeGripper(self, max_it=100, primative=constants.PICK_PRIMATIVE):
    raise NotImplementedError

  @abstractmethod
  def checkGripperClosed(self):
    raise NotImplementedError

  @abstractmethod
  def controlGripper(self, open_ratio, max_it=100):
    raise NotImplementedError

  @abstractmethod
  def _getGripperJointPosition(self):
    raise NotImplementedError

  @abstractmethod
  def _sendPositionCommand(self, commands):
    raise NotImplementedError

  @abstractmethod
  def adjustGripperCommand(self):
    raise NotImplementedError
