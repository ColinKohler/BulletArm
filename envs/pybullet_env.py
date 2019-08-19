import time
import copy
from copy import deepcopy
import numpy as np
import numpy.random as npr

import pybullet as pb
import pybullet_data

from helping_hands_rl_envs.envs.base_env import BaseEnv
from helping_hands_rl_envs.pybullet_toolkit.robots.ur5_rg2 import UR5_RG2
from helping_hands_rl_envs.pybullet_toolkit.robots.kuka import Kuka
import helping_hands_rl_envs.pybullet_toolkit.utils.object_generation as pb_obj_generation

class PyBulletEnv(BaseEnv):
  '''
  PyBullet Arm RL base class.
  '''
  def __init__(self, seed, workspace, max_steps=10, heightmap_size=250, fast_mode=False, render=False,
               action_sequence='pxyr', simulate_grasp=True, pos_candidate=None, perfect_grasp=True, robot='ur5'):
    super(PyBulletEnv, self).__init__(seed, workspace, max_steps, heightmap_size, action_sequence, pos_candidate)

    # Connect to pybullet and add data files to path
    if render:
      self.client = pb.connect(pb.GUI)
    else:
      self.client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.dynamic = not fast_mode

    # Environment specific variables
    self._timestep = 1. / 240.
    if robot == 'ur5':
      self.robot = UR5_RG2()
    elif robot == 'kuka':
      self.robot = Kuka()
    else:
      raise NotImplementedError

    self.block_original_size = 0.05
    self.block_scale_range = (0.6, 0.7)
    self.pick_pre_offset = 0.15
    self.pick_offset = 0.005
    self.place_pre_offset = 0.15
    self.place_offset = self.block_scale_range[1]*self.block_original_size


    # Setup camera parameters
    self.view_matrix = pb.computeViewMatrixFromYawPitchRoll([workspace[0].mean(), workspace[1].mean(), 0], 1.0, -90, -90, 0, 2)
    workspace_x_offset = (workspace[0][1] - workspace[0][0])/2
    workspace_y_offset = (workspace[1][1] - workspace[1][0])/2
    self.proj_matrix = pb.computeProjectionMatrix(-workspace_x_offset, workspace_x_offset, -workspace_y_offset, workspace_y_offset, -1.0, 10.0)

    # Rest pose for arm
    rot = pb.getQuaternionFromEuler([0,np.pi,0])
    self.rest_pose = [[0.0, 0.5, 0.5], rot]

    self.objects = list()
    self.object_types = {}

    self.simulate_grasp = simulate_grasp
    self.perfect_grasp = perfect_grasp

  def reset(self):
    ''''''
    pb.resetSimulation()
    pb.setTimeStep(self._timestep)

    pb.setGravity(0, 0, -10)
    self.table_id = pb.loadURDF('plane.urdf', [0,0,0])

    # Load the UR5 and set it to the home positions
    self.robot.reset()

    # Reset episode vars
    self.objects = list()
    self.object_types = {}

    self.heightmap = None
    self.current_episode_steps = 1

    # Step simulation
    pb.stepSimulation()

    return self._getObservation()

  def saveState(self):
    self.state = {'current_episode_steps': deepcopy(self.current_episode_steps),
                  'objects': deepcopy(self.objects),
                  'env_state': pb.saveState()
                  }
    self.robot.saveState()

  def restoreState(self):
    self.current_episode_steps = self.state['current_episode_steps']
    self.objects = self.state['objects']
    pb.restoreState(self.state['env_state'])
    self.robot.restoreState()

  def takeAction(self, action):
    motion_primative, x, y, z, rot = self._getSpecificAction(action)

    # Get transform for action
    pos = [x, y, z]
    rot_q = pb.getQuaternionFromEuler([0, np.pi, -rot])

    # Take action specfied by motion primative
    if motion_primative == self.PICK_PRIMATIVE:
      if self.perfect_grasp and not self._checkPerfectGrasp(x, y, z, -rot, self.objects):
        return
      self.robot.pick(pos, rot_q, self.pick_pre_offset, dynamic=self.dynamic, objects=self.objects,
                      simulate_grasp=self.simulate_grasp)
    elif motion_primative == self.PLACE_PRIMATIVE:
      if self.robot.holding_obj is not None:
        self.robot.place(pos, rot_q, self.place_pre_offset, dynamic=self.dynamic, simulate_grasp=self.simulate_grasp)
    elif motion_primative == self.PUSH_PRIMATIVE:
      pass
    else:
      raise ValueError('Bad motion primative supplied for action.')

  def isSimValid(self):
    for obj in self.objects:
      if self._isObjectHeld(obj):
        continue
      p = pb_obj_generation.getObjectPosition(obj)
      if not self._isPointInWorkspace(p):
        return False
      if self.pos_candidate is not None:
        if np.abs(self.pos_candidate[0] - p[0]).min() > 0.02 or np.abs(self.pos_candidate[1] - p[1]).min() > 0.02:
          return False
    return True

  def wait(self, iteration):
    if not self.simulate_grasp and self._isHolding():
      return
    for _ in range(iteration):
      pb.stepSimulation()

  def planHouseBuilding1(self, blocks, triangles):
    block_poses = []
    for obj in blocks:
      pos, rot = pb_obj_generation.getObjectPose(obj)
      rot = pb.getEulerFromQuaternion(rot)
      block_poses.append((obj, pos, rot))
    # pick
    if not self._isHolding():
      if not self._checkStack(blocks):
        block_poses.sort(key=lambda x: x[1][-1])
        x, y, z, r = block_poses[0][1][0], block_poses[0][1][1], block_poses[0][1][2] - self.pick_offset, -block_poses[0][2][2]
        for op in block_poses:
          if not self._isObjOnTop(op[0]):
            continue
          x = op[1][0]
          y = op[1][1]
          z = op[1][2] - self.pick_offset
          r = -op[2][2]
          while r < 0:
            r += np.pi
          while r > np.pi:
            r -= np.pi
          break
        return self._encodeAction(self.PICK_PRIMATIVE, x, y, z, r)
      else:
        triangle_pos, triangle_rot = pb_obj_generation.getObjectPose(triangles[0])
        triangle_rot = pb.getEulerFromQuaternion(triangle_rot)
        x = triangle_pos[0]
        y = triangle_pos[1]
        z = triangle_pos[2] - self.pick_offset
        r = -(triangle_rot[2] + np.pi/2)
        while r < 0:
          r += np.pi
        while r > np.pi:
          r -= np.pi
        return self._encodeAction(self.PICK_PRIMATIVE, x, y, z, r)
    # place
    else:
      if self._isObjectHeld(triangles[0]) and not self._checkStack(blocks):
        block_pos = [self._getObjectPosition(o)[:-1] for o in blocks]
        place_pos = self._getValidPositions(self.block_scale_range[1] * self.block_original_size,
                                            self.block_scale_range[1] * self.block_original_size,
                                            block_pos,
                                            1)[0]
        x = place_pos[0]
        y = place_pos[1]
        z = self.place_offset
        r = 0
        return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)

      else:
        block_poses.sort(key=lambda x: x[1][-1], reverse=True)
        x, y, z, r = block_poses[0][1][0], block_poses[0][1][1], block_poses[0][1][2] + self.place_offset, - \
        block_poses[0][2][2]
        for op in block_poses:
          if self._isObjectHeld(op[0]):
            continue
          x = op[1][0]
          y = op[1][1]
          z = op[1][2] + self.place_offset
          r = -op[2][2]
          while r < 0:
            r += np.pi
          while r > np.pi:
            r -= np.pi
          break
        return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)


  def planBlockStacking(self):
    obj_poses = []
    for obj in self.objects:
      pos, rot = pb_obj_generation.getObjectPose(obj)
      rot = pb.getEulerFromQuaternion(rot)
      obj_poses.append((obj, pos, rot))
    # pick
    if not self._isHolding():
      obj_poses.sort(key=lambda x: x[1][-1])
      for op in obj_poses:
        if not self._isObjOnTop(op[0]):
          continue
        x = op[1][0]
        y = op[1][1]
        z = op[1][2] - self.pick_offset
        r = -op[2][2]
        while r < 0:
          r += np.pi
        while r > np.pi:
          r -= np.pi
        return self._encodeAction(self.PICK_PRIMATIVE, x, y, z, r)
    # place
    else:
      obj_poses.sort(key=lambda x: x[1][-1], reverse=True)
      for op in obj_poses:
        if self._isObjectHeld(op[0]):
          continue
        x = op[1][0]
        y = op[1][1]
        z = op[1][2] + self.place_offset
        r = -op[2][2]
        while r < 0:
          r += np.pi
        while r > np.pi:
          r -= np.pi
        return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)

  def _isPointInWorkspace(self, p):
    '''
    Checks if the given point is within the workspace

    Args:
      - p: [x, y, z] point

    Returns: True in point is within workspace, False otherwise
    '''
    return self.workspace[0][0] < p[0] < self.workspace[0][1] and \
           self.workspace[1][0] < p[1] < self.workspace[1][1] and \
           self.workspace[2][0] < p[2] < self.workspace[2][1]

  def _getObservation(self):
    ''''''
    image_arr = pb.getCameraImage(width=self.heightmap_size, height=self.heightmap_size,
                                  viewMatrix=self.view_matrix, projectionMatrix=self.proj_matrix)
    self.heightmap = image_arr[3] - np.min(image_arr[3])
    self.heightmap = self.heightmap.T

    return self._isHolding(), self.heightmap.reshape([self.heightmap_size, self.heightmap_size, 1])

  def _getValidPositions(self, padding, min_distance, existing_positions, num_shapes):
    while True:
      existing_positions_copy = deepcopy(existing_positions)
      valid_positions = []
      for i in range(num_shapes):
        # Generate random drop config
        x_extents = self.workspace[0][1] - self.workspace[0][0]
        y_extents = self.workspace[1][1] - self.workspace[1][0]

        is_position_valid = False
        for j in range(1000):
          if is_position_valid:
            break
          position = [(x_extents - padding) * npr.random_sample() + self.workspace[0][0] + padding / 2,
                      (y_extents - padding) * npr.random_sample() + self.workspace[1][0] + padding / 2]

          if self.pos_candidate is not None:
            position[0] = self.pos_candidate[0][np.abs(self.pos_candidate[0] - position[0]).argmin()]
            position[1] = self.pos_candidate[1][np.abs(self.pos_candidate[1] - position[1]).argmin()]
            if not (self.workspace[0][0]+padding/2 < position[0] < self.workspace[0][1]-padding/2 and
                    self.workspace[1][0]+padding/2 < position[1] < self.workspace[1][1]-padding/2):
              continue

          if existing_positions_copy:
            distances = np.array(list(map(lambda p: np.linalg.norm(np.array(p)-position), existing_positions_copy)))
            is_position_valid = np.all(distances > min_distance)
            # is_position_valid = np.all(np.sum(np.abs(np.array(positions) - np.array(position[:-1])), axis=1) > min_distance)
          else:
            is_position_valid = True
        if is_position_valid:
          existing_positions_copy.append(position)
          valid_positions.append(position)
        else:
          break
      if len(valid_positions) == num_shapes:
        return valid_positions

  def _generateShapes(self, shape_type=0, num_shapes=1, scale=None, pos=None, rot=None,
                           min_distance=0.1, padding=0.2, random_orientation=False):
    ''''''
    if shape_type == self.CUBE:
      min_distance = self.block_original_size * self.block_scale_range[1] * 1.414 * 2
      padding = self.block_original_size * self.block_scale_range[1] * 2
    shape_handles = list()
    positions = [self._getObjectPosition(o)[:-1] for o in self.objects]

    valid_positions = self._getValidPositions(padding, min_distance, positions, num_shapes)

    for position in valid_positions:
      position.append(0.05)
      if random_orientation:
        orientation = pb.getQuaternionFromEuler([0., 0., 2*np.pi*np.random.random_sample()])
      else:
        orientation = pb.getQuaternionFromEuler([0., 0., 0.])
      if not scale:
        scale = npr.uniform(self.block_scale_range[0], self.block_scale_range[1])

      if shape_type == self.CUBE:
        handle = pb_obj_generation.generateCube(position, orientation, scale)
      elif shape_type == self.BRICK:
        handle = pb_obj_generation.generateBrick(position, orientation, scale)
      elif shape_type == self.TRIANGLE:
        handle = pb_obj_generation.generateTriangle(position, orientation, scale)
      else:
        raise NotImplementedError
      shape_handles.append(handle)
    self.objects.extend(shape_handles)
    for h in shape_handles:
      self.object_types[h] = shape_type
    for _ in range(50):
      pb.stepSimulation()
    return shape_handles
    #
    # shape_name = self._getShapeName(shape_type)
    # for i in range(num_shapes):
    #   name = '{}_{}'.format(shape_name, len(shape_handles))
    #
    #   # Generate random drop config
    #   x_extents = self.workspace[0][1] - self.workspace[0][0]
    #   y_extents = self.workspace[1][1] - self.workspace[1][0]
    #
    #   is_position_valid = False
    #   while not is_position_valid:
    #     position = [(x_extents - padding) * npr.random_sample() + self.workspace[0][0] + padding / 2,
    #                 (y_extents - padding) * npr.random_sample() + self.workspace[1][0] + padding / 2,
    #                 0.05]
    #
    #     if self.pos_candidate is not None:
    #       position[0] = self.pos_candidate[0][np.abs(self.pos_candidate[0] - position[0]).argmin()]
    #       position[1] = self.pos_candidate[1][np.abs(self.pos_candidate[1] - position[1]).argmin()]
    #       if not (self.workspace[0][0]+padding/2 < position[0] < self.workspace[0][1]-padding/2 and
    #               self.workspace[1][0]+padding/2 < position[1] < self.workspace[1][1]-padding/2):
    #         continue
    #
    #     if positions:
    #       distances = np.array(list(map(lambda p: np.linalg.norm(np.array(p)-position[:-1]), positions)))
    #       is_position_valid = np.all(distances > min_distance)
    #       # is_position_valid = np.all(np.sum(np.abs(np.array(positions) - np.array(position[:-1])), axis=1) > min_distance)
    #     else:
    #       is_position_valid = True
    #   positions.append(position[:-1])
    #   if random_orientation:
    #     orientation = pb.getQuaternionFromEuler([0., 0., 2*np.pi*np.random.random_sample()])
    #   else:
    #     orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    #
    #   scale = npr.uniform(self.block_scale_range[0], self.block_scale_range[1])
    #
    #   if shape_type == self.CUBE:
    #     handle = pb_obj_generation.generateCube(position, orientation, scale)
    #   elif shape_type == self.BRICK:
    #     handle = pb_obj_generation.generateBrick(position, orientation, scale)
    #   else:
    #     raise NotImplementedError
    #   shape_handles.append(handle)
    #
    # self.objects.extend(shape_handles)
    # for _ in range(50):
    #   pb.stepSimulation()
    # return shape_handles

  def _getObjectPosition(self, obj):
    return pb_obj_generation.getObjectPosition(obj)

  def _isHolding(self):
    return self.robot.holding_obj is not None

  def _getRestPoseMatrix(self):
    T = np.eye(4)
    T[:3, :3] = np.array(pb.getMatrixFromQuaternion(self.rest_pose[1])).reshape((3, 3))
    T[:3, 3] = self.rest_pose[0]
    return T

  def _isObjectHeld(self, obj):
    if obj in self.objects:
      block_position = self._getObjectPosition(obj)
      rest_pose = self._getRestPoseMatrix()
      return block_position[2] > rest_pose[2, -1] - 0.25
    return False

  def _removeObject(self, obj):
    if obj in self.objects:
      # pb.removeBody(obj)
      self._moveObjectOutWorkspace(obj)
      self.robot.openGripper()
      self.objects.remove(obj)

  def _moveObjectOutWorkspace(self, obj):
    pos = [-0.50, 0, 0.25]
    pb.resetBasePositionAndOrientation(obj, pos, pb.getQuaternionFromEuler([0., 0., 0.]))

  def _isObjOnTop(self, obj):
    obj_position = self._getObjectPosition(obj)
    for o in self.objects:
      if self._isObjectHeld(o) or o is obj:
        continue
      block_position = self._getObjectPosition(o)
      if np.allclose(block_position[:-1], obj_position[:-1],
                     atol=self.block_original_size * self.block_scale_range[0] * 2 / 3) and \
          block_position[-1] > obj_position[-1]:
        return False
    return True

  def _getNumTopBlock(self):
    cluster_pos = []
    for obj in self.objects:
      if self._isObjectHeld(obj):
        continue
      block_position = self._getObjectPosition(obj)
      cluster_flag = False
      for cluster in cluster_pos:
        if np.allclose(block_position[:-1], cluster, atol=self.block_original_size*self.block_scale_range[0]*2/3):
          cluster.append(block_position[:-1])
          cluster_flag = True
          break
      if not cluster_flag:
        cluster_pos.append([block_position[:-1]])
    return len(cluster_pos) + self._isHolding()

  def _checkStack(self, objects=None):
    if not objects:
      objects = self.objects
    for obj in objects:
      if self._isObjectHeld(obj):
        return False

    objects = sorted(objects, key=lambda o: self._getObjectPosition(o)[-1])
    for i, obj in enumerate(objects):
      if i == 0:
        continue
      if self.object_types[obj] is self.TRIANGLE:
        if self._getObjectPosition(obj)[-1] - self._getObjectPosition(objects[i-1])[-1] < \
            0.5*self.block_scale_range[0]*self.block_original_size:
          return False
      else:
        if self._getObjectPosition(obj)[-1] - self._getObjectPosition(objects[i-1])[-1] < \
            0.9*self.block_scale_range[0]*self.block_original_size:
          return False
    return True

  def _checkPerfectGrasp(self, x, y, z, rot, objects):
    end_pos = np.array([x, y, z])
    sorted_obj = sorted(objects, key=lambda o: np.linalg.norm(end_pos - pb_obj_generation.getObjectPosition(o)))
    obj_pos, obj_rot = pb_obj_generation.getObjectPose(sorted_obj[0])
    obj_type = self.object_types[sorted_obj[0]]
    obj_rot = pb.getEulerFromQuaternion(obj_rot)
    angle = np.pi - np.abs(np.abs(rot - obj_rot[2]) - np.pi)
    if obj_type is self.CUBE:
      while angle > np.pi / 2:
        angle -= np.pi / 2
      angle = min(angle, np.pi / 2 - angle)
    elif obj_type is self.TRIANGLE:
      angle = abs(angle - np.pi/2)
      angle = min(angle, np.pi - angle)
    return angle < np.pi / 12


  def _checkObjUpright(self, obj):
    triangle_rot = pb_obj_generation.getObjectRotation(obj)
    triangle_rot = pb.getEulerFromQuaternion(triangle_rot)
    return abs(triangle_rot[0]) < 0.1 and abs(triangle_rot[1]) < 0.1

  def _checkOnTop(self, bottom_obj, top_obj):
    # bottom_position = self._getObjectPosition(bottom_obj)
    # top_position = self._getObjectPosition(top_obj)
    # return np.linalg.norm(np.array(bottom_position) - top_position) < 1.8*self.block_scale_range[0]*self.block_original_size and \
    #        top_position[-1] - bottom_position[-1] > 0.9*self.block_scale_range[0]*self.block_original_size
    bottom_position = self._getObjectPosition(bottom_obj)
    top_position = self._getObjectPosition(top_obj)
    if top_position[-1] - bottom_position[-1] < 0.9 * self.block_scale_range[0] * self.block_original_size:
      return False
    contact_points = pb.getContactPoints(top_obj)
    for p in contact_points:
      if p[2] != bottom_obj:
        return False
    return True

  def _getPrimativeHeight(self, motion_primative, x, y):
    '''
    Get the z position for the given action using the current heightmap.
    Args:
      - motion_primative: Pick/place motion primative
      - x: X coordinate for action
      - y: Y coordinate for action
      - offset: How much to offset the action along approach vector
    Returns: Valid Z coordinate for the action
    '''
    x_pixel, y_pixel = self._getPixelsFromPos(x, y)
    local_region = self.heightmap[int(max(y_pixel - self.heightmap_size/20, 0)):int(min(y_pixel + self.heightmap_size/20, self.heightmap_size)), \
                                  int(max(x_pixel - self.heightmap_size/20, 0)):int(min(x_pixel + self.heightmap_size/20, self.heightmap_size))]
    try:
      safe_z_pos = np.max(local_region) + self.workspace[2][0]
    except ValueError:
      safe_z_pos = self.workspace[2][0]
    if motion_primative == self.PICK_PRIMATIVE:
      safe_z_pos -= self.pick_offset
      safe_z_pos = max(safe_z_pos, 0.025)
    else:
      safe_z_pos += self.place_offset
    return safe_z_pos
