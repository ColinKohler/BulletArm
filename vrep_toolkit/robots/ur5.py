import os
import sys
import time
import numpy as np

from vrep_arm_toolkit.simulation import vrep
import vrep_arm_toolkit.utils.vrep_utils as utils
from vrep_arm_toolkit.utils import transformations

class UR5(object):
  '''
  VRep UR5 robot class. Works with any gripper included in this package.
  Currently only does IK control.

  Args:
    - sim_client: VRep client object to communicate with simulator over
    - gripper: Gripper which is attached to UR5 in simulator. Must be included in 'grippers'
    '''
  def __init__(self, sim_client, gripper):
    self.sim_client = sim_client
    self.gripper = gripper

    # Create handles to the UR5 target and tip which control IK control
    sim_ret, self.UR5_target = utils.getObjectHandle(self.sim_client, 'UR5_target')
    sim_ret, self.gripper_tip = utils.getObjectHandle(self.sim_client, 'UR5_tip')

  def getEndEffectorPose(self):
    '''
    Get the current end effector pose

    Returns: 4D pose of the gripper
    '''
    sim_ret, pose = utils.getObjectPose(self.sim_client, self.gripper_tip)
    return pose

  def openGripper(self):
    '''
    Opens the gripper as much as is possible
    '''
    self.gripper.open()

  def closeGripper(self):
    '''
    Closes the gripper as much as is possible

    Returns: True if gripper is fully closed, False otherwise
    '''
    return self.gripper.close()

  def moveTo(self, pose, move_step_size=0.01, single_step=False):
    '''
    Moves the tip of the gripper to the target pose

    Args:
      - pose: 4D target pose
    '''
    # Get current position and orientation of UR5 target
    sim_ret, UR5_target_position = utils.getObjectPosition(self.sim_client, self.UR5_target)
    sim_ret, UR5_target_orientation = utils.getObjectOrientation(self.sim_client, self.UR5_target)

    # Calculate the movement increments
    move_direction = pose[:3,-1] - UR5_target_position
    move_magnitude = np.linalg.norm(move_direction)
    move_step = move_step_size * move_direction / move_magnitude
    num_move_steps = int(np.ceil(move_magnitude / move_step_size))

    # Calculate the rotation increments
    rotation = np.asarray(transformations.euler_from_matrix(pose))
    rotation_step = rotation - UR5_target_orientation
    rotation_step[rotation >= 0] = 0.1
    rotation_step[rotation < 0] = -0.1
    num_rotation_steps = np.ceil((rotation - UR5_target_orientation) / rotation_step).astype(np.int)

    # Move and rotate to the target pose
    if not single_step:
      for i in range(max(num_move_steps, np.max(num_rotation_steps))):
        pos = UR5_target_position + move_step*min(i, num_move_steps)
        rot = [UR5_target_orientation[0]+rotation_step[0]*min(i, num_rotation_steps[0]),
               UR5_target_orientation[1]+rotation_step[1]*min(i, num_rotation_steps[1]),
               UR5_target_orientation[2]+rotation_step[2]*min(i, num_rotation_steps[2])]
        utils.setObjectPosition(self.sim_client, self.UR5_target, pos)
        utils.setObjectOrientation(self.sim_client, self.UR5_target, rot)

    utils.setObjectPosition(self.sim_client, self.UR5_target, pose[:3,-1])
    utils.setObjectOrientation(self.sim_client, self.UR5_target, rotation)


  def pick(self, grasp_pose, offset, fast_mode=False):
    '''
    Attempts to execute a pick at the target pose

    Args:
      - grasp_pose: 4D pose to execture grasp at
      - offset: Grasp offset for pre-grasp pose
      - fast_mode: Teleport the arm when it doesn't interact with other objects.

    Returns: True if pick was successful, False otherwise
    '''
    pre_grasp_pose = np.copy(grasp_pose)
    # TODO: offset should be along approach vector. Currently just the z component.
    pre_grasp_pose[2,-1] += offset

    self.openGripper()
    self.moveTo(pre_grasp_pose, single_step=fast_mode)
    self.moveTo(grasp_pose)
    is_fully_closed = self.closeGripper()
    if is_fully_closed:
      self.moveTo(pre_grasp_pose, single_step=fast_mode)
    else:
      self.moveTo(pre_grasp_pose, single_step=False)
    is_fully_closed = self.closeGripper()

    return not is_fully_closed

  def place(self, drop_pose, offset, fast_mode=False):
    '''
    Attempts to execute a place at the target pose

    Args:
      - drop_pose: 4D pose to place object at
      - offset: Grasp offset for pre-grasp pose
      - fast_mode: Teleport the arm when it doesn't interact with other objects.
    '''
    pre_drop_pose = np.copy(drop_pose)
    pre_drop_pose[2,-1] += offset

    self.moveTo(pre_drop_pose)
    self.moveTo(drop_pose)
    self.openGripper()
    self.moveTo(pre_drop_pose, single_step=fast_mode)
