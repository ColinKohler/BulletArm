'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import copy
import time
import math
import numpy as np
import pybullet as pb
from collections import deque
from scipy.ndimage import rotate

from bulletarm.pybullet.utils import pybullet_util
from bulletarm.pybullet.utils import constants
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

    self.state = {
      'holding_obj': None,
      'gripper_closed': False
    }

    self.position_gain = 1.0
    self.speed = 1e-4
    self.adjust_gripper_after_lift = False
    self.force_history = np.zeros((64, 6)).tolist()
    self.zero_force = None

  def saveState(self):
    '''
    Set the robot state. The state consists of the object that is being held and
    if the gripper is closed.
    '''
    self.state = {
      'holding_obj': self.gripper.holding_obj,
      'gripper_closed': self.gripper.closed
    }

  def restoreState(self):
    '''
    Restores the robot to the previously saved state.
    '''
    self.gripper.holding_obj = self.state['holding_obj']
    self.gripper.closed = self.state['gripper_closed']
    if self.gripper_closed:
      self.closeGripper(max_it=0)
    else:
      self.openGripper()

  def initialize(self):
    ''''''
    self.gripper.initialize(self.id)

    pb.resetBasePositionAndOrientation(self.id, [-0.2,0,0], [0,0,0,1])

    self.num_joints = pb.getNumJoints(self.id)
    [pb.resetJointState(self.id, idx, self.home_positions[idx]) for idx in range(self.num_joints)]
    pb.enableJointForceTorqueSensor(self.id, self.wrist_index)
    for j in range(pb.getNumJoints(self.id)):
      pb.changeDynamics(self.id, j, linearDamping=0, angularDamping=0)

    self.arm_joint_names = list()
    self.arm_joint_indices = list()
    for i in range (self.num_joints):
      joint_info = pb.getJointInfo(self.id, i)
      if i in range(self.num_dofs):
        self.arm_joint_names.append(str(joint_info[1]))
        self.arm_joint_indices.append(i)

    pb.changeDynamics(
      self.id,
      self.finger_idxs[0],
      lateralFriction=1.0,
      spinningFriction=0.001,
    )
    pb.changeDynamics(
      self.id,
      self.finger_idxs[1],
      lateralFriction=1.0,
      spinningFriction=0.001,
    )

    self.force_history = np.zeros((64, 6)).tolist()

    # Zero force out
    pb.stepSimulation()
    wrist_force, wrist_moment = self.getWristForce()
    self.zero_force = np.concatenate((wrist_force, wrist_moment))

  def reset(self):
    ''''''
    self.gripper.reset()
    self.moveToJ(self.home_positions_joint[:self.num_dofs], dynamic=False)

    self.force_history = np.zeros((64, 6)).tolist()

    # Zero force out
    pb.stepSimulation()
    wrist_force, wrist_moment = self.getWristForce()
    self.zero_force = np.concatenate((wrist_force, wrist_moment))

  def getHeldObject(self):
    '''
    '''
    return self.gripper.holding_obj

  def getPickedObj(self, objects):
    '''
    Get the object which is currently being held by the gripper.

    Args:
      objects (numpy.array): Objects to check if are being held.

    Returns:
      (pybullet.objects.PybulletObject): Object being held.
    '''
    self.gripper.getPickedObj(objects)
    return self.gripper.holding_obj

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
    self.gripper.open()

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
      gripper_fully_closed = self.gripper.close()
      if gripper_fully_closed:
        self.gripper.open()

      # Adjust gripper command after moving to pre-grasp pose. Useful in cluttered domains.
      # This will increase grasp chance but gripper will shift while lifting object.
      if self.adjust_gripper_after_lift:
        self.moveTo(pre_pos, pre_rot, True)
        self.gripper.adjustCommand()
      # Adjust gripper command before moving to pre-grasp pose.
      # This will increase gripper stabilization but will reduce grasp chance.
      else:
        self.gripper.adjustCommand()
        self.moveTo(pre_pos, pre_rot, True)

      for i in range(100):
        pb.stepSimulation()
    else:
      self.moveTo(pos, rot, dynamic)

    self.gripper.getPickedObj(objects)
    self.moveToJ(self.home_positions_joint, dynamic)
    self.gripper.checkClosed()

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
    self.gripper.open()
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

    self.gripper.close()
    self.moveTo(pre_pos, rot, dynamic)
    self.moveTo(pos, rot, True)
    self.moveTo(goal_pos, rot, True)
    self.gripper.open()
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
    self.gripper.close()
    # for mid in np.arange(0, offset, 0.05)[1:]:
    #   self.moveTo(pos + m[:, 2] * mid, rot, True)
    self.moveTo(pre_pos, rot, True)
    self.gripper.open()
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
      else:
        m_prime = m.dot(transformations.euler_matrix(0, -theta, 0))
      waypoint_rot.append(transformations.quaternion_from_matrix(m_prime))

    self.moveTo(pre_pos, rot, dynamic)
    self.moveTo(pos, rot, True)
    self.gripper.close()
    for i in range(len(waypoint_theta)):
      self.moveTo(waypoint_pos[i], waypoint_rot[i], True)

    self.gripper.open()
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
    if dynamic or not self.gripper.holding_obj:
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
    if dynamic or not self.gripper.holding_obj:
      self._moveToJointPose(joint_pose, dynamic)
    else:
      self._teleportArmWithObjJointPose(joint_pose)

  def _moveToJointPose(self, target_pose, dynamic=True):
    '''
    Move the desired joint positions

    Args:
      joint_pose (numpy.array): Joint positions for each joint in the manipulator.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
    '''
    if dynamic:
      t0 = time.time()
      i = 0
      past_joint_pos = deque(maxlen=5)
      while (time.time() - t0) < 1.:
        # Calculate difference between current pose and target pose
        joint_state = pb.getJointStates(self.id, self.arm_joint_indices)
        joint_pos = np.array(list(zip(*joint_state))[0])
        target_pose = np.array(target_pose)
        diff = target_pose - joint_pos

        # Exit if the difference is within a small tolerance
        if all(np.abs(diff) < 5e-3):
          return
        # Exit if the robot is stuck
        #if (len(past_joint_pos) == 5 and np.allclose(past_joint_pos[-1], past_joint_pos, atol=5e-3)):
        #  return

        # Move with constant velocity
        norm = np.linalg.norm(diff)
        v = diff / norm if norm > 0 else 0
        step = joint_pos + v * self.speed
        self._sendPositionCommand(step)
        pb.stepSimulation()

        # Read force sensor
        wrist_force, wrist_moment = self.getWristForce()
        force = np.concatenate((wrist_force, wrist_moment))
        force[2] -= self.zero_force[2]
        force[5] -= self.zero_force[5]
        self.force_history.append(force)

        past_joint_pos.append(joint_pos)
        i += 1
    else:
      self._setJointPoses(target_pose)

  def _moveToCartesianPose(self, pos, rot, dynamic=True, pos_th=1e-4, rot_th=1e-4):
    '''
    Move the end effector to the desired cartesian pose.

    Args:
      pos (numpy.array): Desired end-effector position.
      rot (numpy.array): Desired end-effector orientation.
      dynamic (bool): Simualte arm dynamics when moving the arm. Defaults to True.
    '''
    ik_solve = self._calculateIK(pos, rot)
    self._moveToJointPose(ik_solve, dynamic)

  def _teleportArmWithObj(self, pos, rot):
    '''
    Teleport the arm to the given pose along with the object that is being grasped.

    Args:
      pos (numpy.array): Desired end-effector position.
      rot (numpy.array): Desired end-effector orientation.
    '''
    if not self.gripper.holding_obj:
      self._moveToCartesianPose(pos, rot, False)
      return

    end_pos = self._getEndEffectorPosition()
    end_rot = self._getEndEffectorRotation()
    obj_pos, obj_rot = self.gripper.holding_obj.getPose()
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

    self.gripper.holding_obj.resetPose(obj_pos_, obj_rot_)

  def getEndToHoldingObj(self):
    '''
    Get the transfomration from the end effector to the object currently beinging held.

    Returns:
      numpy.array: Transformation matrix
    '''
    if not self.gripper.holding_obj:
      return np.zeros((4, 4))

    end_pos = self._getEndEffectorPosition()
    end_rot = self._getEndEffectorRotation()
    obj_pos, obj_rot = self.gripper.holding_obj.getPose()
    oTend = pybullet_util.getMatrix(end_pos, end_rot)
    oTobj = pybullet_util.getMatrix(obj_pos, obj_rot)
    endTobj = np.linalg.inv(oTend).dot(oTobj)

    return endTobj

  def getWristForce(self):
    wrist_info = list(pb.getJointState(self.id, self.wrist_index)[2])
    wrist_force = np.array(wrist_info[:3])
    wrist_moment = np.array(wrist_info[3:])

    # Transform to world frame
    wrist_rot = pb.getMatrixFromQuaternion(pb.getLinkState(self.id, self.wrist_index)[5])
    wrist_rot = np.array(list(wrist_rot)).reshape((3,3))
    wrist_force = np.dot(wrist_rot, wrist_force)
    wrist_moment = np.dot(wrist_rot, wrist_moment)

    return wrist_force, wrist_moment

  def gripperHasForce(self):
    return pb.getJointState(self.id, self.wrist_idx)[3] >= 2

  def getGripperImg(self, img_size, workspace_size, obs_size_m):
    gripper_state = self.gripper.getOpenRatio()
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


  def _teleportArmWithObjJointPose(self, joint_pose):
    '''
    Directly set the arm to the given joint positions. Moves the object being held.

    Args:
      joint_pose (numpy.array): The joint positions for the manipulators
    '''
    if not self.gripper.holding_obj:
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

    self.gripper.holding_obj.resetPose(obj_pos_, obj_rot_)

  def _getEndEffectorPose(self):
    '''
    Get the current end effector pose.

    Returns:
      (numpy.array): The end effector pose.
    '''
    state = pb.getLinkState(self.id, self.end_effector_index)
    return state

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

  def _calculateIK(self, pos, rot):
    joints = pb.calculateInverseKinematics(
      bodyUniqueId=self.id,
      endEffectorLinkIndex=self.end_effector_index,
      targetPosition=pos,
      targetOrientation=rot,
      lowerLimits=self.ll,
      upperLimits=self.ul,
      jointRanges=self.jr,
      restPoses=self.ml,
      maxNumIterations=100,
      residualThreshold=1e-5
    )
    joints = np.float32(joints)
    return joints[:self.num_dofs]

  def _sendPositionCommand(self, commands):
    ''''''
    num_motors = len(self.arm_joint_indices)
    pb.setJointMotorControlArray(
      bodyIndex=self.id,
      jointIndices=self.arm_joint_indices,
      controlMode=pb.POSITION_CONTROL,
      targetPositions=commands,
      forces=self.max_torque,
      positionGains=[self.position_gain]*num_motors,
    )

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
