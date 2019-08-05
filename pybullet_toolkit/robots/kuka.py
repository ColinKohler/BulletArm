import os
import copy
import math
import numpy as np
import numpy.random as npr
from collections import deque

import pybullet as pb
import pybullet_data

import helping_hands_rl_envs
import time

from helping_hands_rl_envs.pybullet_toolkit.utils import pybullet_util
from helping_hands_rl_envs.pybullet_toolkit.utils import object_generation
from helping_hands_rl_envs.pybullet_toolkit.utils import transformations

class Kuka(object):
  '''

  '''
  def __init__(self):
    # Setup arm and gripper variables
    self.max_velocity = .35
    self.max_force = 200.
    self.finger_a_force = 2
    self.finger_b_force = 2.5
    self.finger_tip_force = 2
    self.end_effector_index = 14
    self.gripper_index = 7

    # lower limits for null space
    self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
    # upper limits for null space
    self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    # joint ranges for null space
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    # restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    # joint damping coefficents
    self.jd = [
      0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
      0.00001, 0.00001, 0.00001, 0.00001
    ]

    self.home_positions = [0.3926, 0., -2.137, 1.432, 0, -1.591, 0.071, 0., 0., 0., 0., 0., 0., 0., 0.]
    # self.home_positions = [
    #     0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
    #     -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
    # ]

    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)

    self.gripper_joint_limit = [0, 0.3]

    self.holding_obj = None
    self.gripper_closed = False
    self.state = {
      'holding_obj': self.holding_obj,
      'gripper_closed': self.gripper_closed
    }


  def reset(self):
    ''''''
    ur5_urdf_filepath = os.path.join(self.root_dir, 'urdf/kuka/kuka_with_gripper2.sdf')
    self.id = pb.loadSDF(ur5_urdf_filepath)[0]
    pb.resetBasePositionAndOrientation(self.id, [-0.2,0,0], [0,0,0,1])

    # self.is_holding = False
    self.gripper_closed = False
    self.holding_obj = None
    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]
    self.openGripper()
    # self.gripper_joint_names = list()
    # self.gripper_joint_indices = list()
    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(7):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)
      # elif i in range(10, 12):
      #   self.gripper_joint_names.append(str(joint_info[1]))
      #   self.gripper_joint_indices.append(i)

  def saveState(self):
    self.state = {
      'holding_obj': self.holding_obj,
      'gripper_closed': self.gripper_closed
    }

  def restoreState(self):
    self.holding_obj = self.state['holding_obj']
    self.gripper_closed = self.state['gripper_closed']
    if self.gripper_closed:
      self._sendGripperCloseCommand()
    else:
      self._sendGripperOpenCommand()

  def pick(self, pos, rot, offset, dynamic=True, objects=None, simulate_grasp=True, perfect_grasp=False):
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
      if perfect_grasp and not self._checkPerfectGrasp(objects):
        self.moveTo(pre_pos, pre_rot, dynamic)
      else:
        # Grasp object and lift up to pre pose
        gripper_fully_closed = self.closeGripper()
        if gripper_fully_closed:
          self.openGripper()
          self.moveTo(pre_pos, pre_rot, dynamic)
        else:
          self.moveTo(pre_pos, pre_rot, True)
          self.holding_obj = self.getPickedObj(objects)

    else:
      self.moveTo(pos, rot, dynamic)
      self.holding_obj = self.getPickedObj(objects)

    self.moveToJ(self.home_positions[:7], dynamic)
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
    self.moveToJ(self.home_positions[:7], dynamic)

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

  def getPickedObj(self, objects):
    if not objects:
      return None
    end_pos = self._getEndEffectorPosition()
    sorted_obj = sorted(objects, key=lambda o: np.linalg.norm(end_pos-object_generation.getObjectPosition(o)))
    obj_pos = object_generation.getObjectPosition(sorted_obj[0])
    if np.linalg.norm(end_pos[:-1]-obj_pos[:-1]) < 0.05 and np.abs(end_pos[-1]-obj_pos[-1]) < 0.025:
      return sorted_obj[0]
    # if object_generation.getObjectPosition(sorted_obj[0])[-1] > 0.25:
    #   return sorted_obj[0]
    return None

  def closeGripper(self):
    ''''''
    p1, p2 = self._getGripperJointPosition()
    target = self.gripper_joint_limit[0]
    self._sendGripperCommand(target)
    self.gripper_closed = True
    it = 0
    while abs(target-p1) + abs(target-p2) > 0.001:
    # while p1 < 0.036:
      pb.stepSimulation()
      it += 1
      if it > 100:
        return False
      p1_, p2_ = self._getGripperJointPosition()
      if p1 <= p1_ and p2 <= p2_:
        return False
      p1 = p1_
      p2 = p2_
    return True

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
    self._sendGripperCommand(target)
    self.gripper_closed = False
    self.holding_obj = None
    it = 0
    while abs(target-p1) + abs(target-p2) > 0.001:
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
    max_outer_it = 100
    max_inner_it = 1000

    while not close_enough and outer_it < max_outer_it:
      ik_solve = pb.calculateInverseKinematics(self.id, self.end_effector_index, pos, rot,
                                               lowerLimits=self.ll,
                                               upperLimits=self.ul,
                                               jointRanges=self.jr,
                                               restPoses=self.rp
                                               )[:7]
      self._moveToJointPose(ik_solve, dynamic, max_inner_it)

      ls = pb.getLinkState(self.id, self.end_effector_index)
      new_pos = list(ls[4])
      new_rot = list(ls[5])
      close_enough = np.allclose(np.array(new_pos + new_rot), np.array(list(pos) + list(rot)), atol=threshold)
      outer_it += 1

  def _teleportArmWithObj(self, pos, rot):
    if not self.holding_obj:
      self._moveToCartesianPose(pos, rot, False)
      return

    end_pos = self._getEndEffectorPosition()
    end_rot = self._getEndEffectorRotation()
    obj_pos, obj_rot = object_generation.getObjectPose(self.holding_obj)
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

    pb.resetBasePositionAndOrientation(self.holding_obj, obj_pos_, obj_rot_)

  def _teleportArmWithObjJointPose(self, joint_pose):
    if not self.holding_obj:
      self._moveToJointPose(joint_pose, False)
      return

    end_pos = self._getEndEffectorPosition()
    end_rot = self._getEndEffectorRotation()
    obj_pos, obj_rot = object_generation.getObjectPose(self.holding_obj)
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

    pb.resetBasePositionAndOrientation(self.holding_obj, obj_pos_, obj_rot_)

  def _getEndEffectorPosition(self):
    ''''''
    state = pb.getLinkState(self.id, self.end_effector_index)
    return np.array(state[4])

  def _getEndEffectorRotation(self):
    state = pb.getLinkState(self.id, self.end_effector_index)
    return np.array(state[5])

  def _getGripperJointPosition(self):
    p1 = -pb.getJointState(self.id, 8)[0]
    p2 = pb.getJointState(self.id, 11)[0]
    return p1, p2

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    # pb.setJointMotorControlArray(self.id, self.arm_joint_indices, pb.POSITION_CONTROL, commands,
    #                              targetVelocities=[0.]*num_motors,
    #                              forces=[self.max_force]*num_motors,
    #                              positionGains=[0.3]*num_motors,
    #                              velocityGains=[1.0]*num_motors)
    for i in range(num_motors):
      pb.setJointMotorControl2(bodyUniqueId=self.id,
                               jointIndex=i,
                               controlMode=pb.POSITION_CONTROL,
                               targetPosition=commands[i],
                               targetVelocity=0,
                               force=self.max_force,
                               maxVelocity=self.max_velocity,
                               positionGain=0.3,
                               velocityGain=1)

  def _sendGripperCommand(self, target_pos):
    pb.setJointMotorControl2(self.id,
                             7,
                             pb.POSITION_CONTROL,
                             targetPosition=target_pos,
                             force=self.max_force)
    pb.setJointMotorControl2(self.id,
                             8,
                             pb.POSITION_CONTROL,
                             targetPosition=-target_pos,
                             force=self.finger_a_force)
    pb.setJointMotorControl2(self.id,
                             11,
                             pb.POSITION_CONTROL,
                             targetPosition=target_pos,
                             force=self.finger_b_force)

    pb.setJointMotorControl2(self.id,
                             10,
                             pb.POSITION_CONTROL,
                             targetPosition=0,
                             force=self.finger_tip_force)
    pb.setJointMotorControl2(self.id,
                             13,
                             pb.POSITION_CONTROL,
                             targetPosition=0,
                             force=self.finger_tip_force)

  def _sendGripperCloseCommand(self):
    target_pos = self.gripper_joint_limit[0]
    # pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
    #                              targetPositions=[target_pos, target_pos], forces=self.gripper_close_force)

    pb.setJointMotorControl2(self.id,
                            7,
                            pb.POSITION_CONTROL,
                            targetPosition=target_pos,
                            force=self.max_force)
    pb.setJointMotorControl2(self.id,
                            8,
                            pb.POSITION_CONTROL,
                            targetPosition=-target_pos,
                            force=self.finger_a_force)
    pb.setJointMotorControl2(self.id,
                            11,
                            pb.POSITION_CONTROL,
                            targetPosition=target_pos,
                            force=self.finger_b_force)

    pb.setJointMotorControl2(self.id,
                            10,
                            pb.POSITION_CONTROL,
                            targetPosition=0,
                            force=self.finger_tip_force)
    pb.setJointMotorControl2(self.id,
                            13,
                            pb.POSITION_CONTROL,
                            targetPosition=0,
                            force=self.finger_tip_force)

  def _sendGripperOpenCommand(self):
    target_pos = self.gripper_joint_limit[1]
    # pb.setJointMotorControlArray(self.id, self.gripper_joint_indices, pb.POSITION_CONTROL,
    #                              targetPositions=[target_pos, target_pos], forces=self.gripper_open_force)
    pb.setJointMotorControl2(self.id,
                            7,
                            pb.POSITION_CONTROL,
                            targetPosition=target_pos,
                            force=self.max_force)
    pb.setJointMotorControl2(self.id,
                            8,
                            pb.POSITION_CONTROL,
                            targetPosition=-target_pos,
                            force=self.finger_a_force)
    pb.setJointMotorControl2(self.id,
                            11,
                            pb.POSITION_CONTROL,
                            targetPosition=target_pos,
                            force=self.finger_b_force)

    pb.setJointMotorControl2(self.id,
                            10,
                            pb.POSITION_CONTROL,
                            targetPosition=0,
                            force=self.finger_tip_force)
    pb.setJointMotorControl2(self.id,
                            13,
                            pb.POSITION_CONTROL,
                            targetPosition=0,
                            force=self.finger_tip_force)

  def _setJointPoses(self, q_poses):
    ''''''
    for i in range(len(q_poses)):
      motor = self.arm_joint_indices[i]
      pb.resetJointState(self.id, motor, q_poses[i])

    self._sendPositionCommand(q_poses)

  def _checkPerfectGrasp(self, objects):
    if not objects:
      return False
    end_pos = self._getEndEffectorPosition()
    end_rot = transformations.euler_from_quaternion(self._getEndEffectorRotation())
    sorted_obj = sorted(objects, key=lambda o: np.linalg.norm(end_pos - object_generation.getObjectPosition(o)))
    obj_pos, obj_rot = object_generation.getObjectPose(sorted_obj[0])
    obj_rot = transformations.euler_from_quaternion(obj_rot)
    angle = np.pi - np.abs(np.abs(end_rot[2] - obj_rot[2]) - np.pi)
    while angle > np.pi/2:
      angle -= np.pi/2
    angle = min(angle, np.pi/2-angle)
    return angle < np.pi/15

