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
from helping_hands_rl_envs.pybullet_toolkit.robots.ur5_robotiq import UR5_Robotiq
import helping_hands_rl_envs.pybullet_toolkit.utils.object_generation as pb_obj_generation

import pickle
import os

class NoValidPositionException(Exception):
  pass

class PyBulletEnv(BaseEnv):
  '''
  PyBullet Arm RL base class.
  '''
  def __init__(self, config):
    if 'robot' not in config:
      config['robot'] = 'ur5'
    if 'pos_candidate' not in config:
      config['pos_candidate'] = None
    if 'perfect_grasp' not in config:
      config['perfect_grasp'] = False
    if 'perfect_place' not in config:
      config['perfect_place'] = False
    seed = config['seed']
    workspace = config['workspace']
    max_steps = config['max_steps']
    obs_size = config['obs_size']
    fast_mode = config['fast_mode']
    render = config['render']
    action_sequence = config['action_sequence']
    simulate_grasp = config['simulate_grasp']
    pos_candidate = config['pos_candidate']
    perfect_grasp = config['perfect_grasp']
    perfect_place = config['perfect_place']
    robot = config['robot']
    super(PyBulletEnv, self).__init__(seed, workspace, max_steps, obs_size, action_sequence, pos_candidate)

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
    elif robot == 'ur5_robotiq':
      self.robot = UR5_Robotiq()
    elif robot == 'kuka':
      self.robot = Kuka()
    else:
      raise NotImplementedError

    self.block_original_size = 0.05
    self.block_scale_range = (0.6, 0.7)
    self.min_block_size = self.block_original_size * self.block_scale_range[0]
    self.max_block_size = self.block_original_size * self.block_scale_range[1]
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
    self.perfect_place = perfect_place

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
                  'env_state': pb.saveState(),
                  'heightmap': deepcopy(self.heightmap)
                  }
    self.robot.saveState()

  def restoreState(self):
    self.current_episode_steps = self.state['current_episode_steps']
    self.objects = self.state['objects']
    self.heightmap = self.state['heightmap']
    pb.restoreState(self.state['env_state'])
    self.robot.restoreState()

  def saveEnvToFile(self, path):
    bullet_file = os.path.join(path, 'env.bullet')
    pickle_file = os.path.join(path, 'env.pickle')
    pb.saveBullet(bullet_file)
    self.robot.saveState()
    state = {
      'heightmap': deepcopy(self.heightmap),
      'current_episode_steps': deepcopy(self.current_episode_steps),
      'objects': deepcopy(self.objects),
      'robot_state': deepcopy(self.robot.state),
      'random_state': np.random.get_state()
    }
    with open(pickle_file, 'wb') as f:
      pickle.dump(state, f)

  def loadEnvFromFile(self, path):
    bullet_file = os.path.join(path, 'env.bullet')
    pickle_file = os.path.join(path, 'env.pickle')
    pb.restoreState(fileName=bullet_file)
    with open(pickle_file, 'rb') as f:
      state = pickle.load(f)
    self.heightmap = state['heightmap']
    self.current_episode_steps = state['current_episode_steps']
    self.objects = state['objects']
    self.robot.state = state['robot_state']
    np.random.set_state(state['random_state'])
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
        if self.perfect_place and not self._checkPerfectPlace(x, y, z, -rot, self.objects):
          return
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
    [pb.stepSimulation() for _ in range(iteration)]

  def getPickingBlockPlan(self, blocks, second_biggest=False):
    block_poses = []
    for obj in blocks:
      pos, rot = pb_obj_generation.getObjectPose(obj)
      rot = pb.getEulerFromQuaternion(rot)
      block_poses.append((obj, pos, rot))

    if second_biggest:
      block_poses.sort(key=lambda x: x[1][-1], reverse=True)
      block_poses = block_poses[1:] + block_poses[:1]

    x, y, z, r = block_poses[0][1][0], block_poses[0][1][1], block_poses[0][1][2]-self.pick_offset, block_poses[0][2][2]
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

  def getStackingBlockPlan(self, blocks):
    block_poses = []
    for obj in blocks:
      pos, rot = pb_obj_generation.getObjectPose(obj)
      rot = pb.getEulerFromQuaternion(rot)
      block_poses.append((obj, pos, rot))
    block_poses.sort(key=lambda x: x[1][-1], reverse=True)
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
      return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)

  def getPickingRoofPlan(self, roofs):
    roof_pos, roof_rot = pb_obj_generation.getObjectPose(roofs[0])
    roof_rot = pb.getEulerFromQuaternion(roof_rot)
    x = roof_pos[0]
    y = roof_pos[1]
    z = roof_pos[2] - self.pick_offset
    r = -(roof_rot[2] + np.pi / 2)
    while r < 0:
      r += np.pi
    while r > np.pi:
      r -= np.pi
    return self._encodeAction(self.PICK_PRIMATIVE, x, y, z, r)

  def getPickingBrickPlan(self, bricks):
    return self.getPickingRoofPlan(bricks)

  def planHouseBuilding1(self, blocks, triangles):
    # pick
    if not self._isHolding():
      # blocks not stacked, pick block
      if not self._checkStack(blocks):
        return self.getPickingBlockPlan(blocks, True)
      # blocks stacked, pick triangle
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
      # holding triangle, but block not stacked, put down triangle
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
      # stack on block
      else:
        return self.getStackingBlockPlan(blocks)

  def blockPosValidHouseBuilding2(self, blocks):
    block1_pos = self._getObjectPosition(blocks[0])
    block2_pos = self._getObjectPosition(blocks[1])
    max_block_size = self.max_block_size
    dist = np.linalg.norm(np.array(block1_pos) - np.array(block2_pos))
    return block1_pos[-1] < self.max_block_size and block2_pos[-1] < self.max_block_size and dist < 2.2 * max_block_size

  def brickPosValidHouseBuilding3(self, blocks, bricks):
    return self._checkOnTop(blocks[0], bricks[0]) and \
           self._checkOnTop(blocks[1], bricks[0]) and \
           self._checkInBetween(bricks[0], blocks[0], blocks[1])

  def planHouseBuilding2(self, blocks, roofs):
    block1_pos = self._getObjectPosition(blocks[0])
    block2_pos = self._getObjectPosition(blocks[1])
    max_block_size = self.max_block_size
    dist = np.linalg.norm(np.array(block1_pos) - np.array(block2_pos))
    def dist_valid(d):
      return 1.5 * max_block_size < d < 2 * max_block_size
    valid_block_pos = dist_valid(dist)
    # not holding, do pick
    if not self._isHolding():
      # block pos not valid, adjust block pos => pick block
      if not valid_block_pos:
        return self.getPickingBlockPlan(blocks)
      # block pos valid, pick roof
      else:
        return self.getPickingRoofPlan(roofs)
    # holding, do placing
    else:
      if self._isObjectHeld(roofs[0]):
        # holding roof, but block pos not valid => place roof on arbitrary pos
        if not valid_block_pos:
          block_pos = [self._getObjectPosition(o)[:-1] for o in blocks]
          try:
            place_pos = self._getValidPositions(self.max_block_size * 3,
                                                self.max_block_size * 2,
                                                block_pos,
                                                1)[0]
          except NoValidPositionException:
            place_pos = self._getValidPositions(self.max_block_size * 3,
                                                self.max_block_size * 2,
                                                [],
                                                1)[0]
          x, y, z, r = place_pos[0], place_pos[1], self.place_offset, 0
          return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)
        # holding roof, block pos valid => place roof on top
        else:
          block_pos = [self._getObjectPosition(o) for o in blocks]
          middle_point = np.mean((np.array(block_pos[0]), np.array(block_pos[1])), axis=0)
          x, y, z = middle_point[0], middle_point[1], middle_point[2]+self.place_offset
          slop = (block_pos[0][1] - block_pos[1][1]) / (block_pos[0][0] - block_pos[1][0])
          r = -np.arctan(slop)-np.pi/2
          while r < 0:
            r += np.pi
          return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)
      # holding block, place block on valid pos
      else:
        place_pos = self._getValidPositions(self.max_block_size * 2, self.max_block_size * 2, [], 1)[0]
        for i in range(10000):
          if self._isObjectHeld(blocks[0]):
            other_block = blocks[1]
          else:
            other_block = blocks[0]
          other_block_pos = self._getObjectPosition(other_block)
          roof_pos = [self._getObjectPosition(roofs[0])[:-1]]
          try:
            place_pos = self._getValidPositions(self.max_block_size * 2,
                                                self.max_block_size * 2,
                                                roof_pos,
                                                1)[0]
          except NoValidPositionException:
            continue
          dist = np.linalg.norm(np.array(other_block_pos[:-1]) - np.array(place_pos))
          if dist_valid(dist):
            break
        x, y, z, r = place_pos[0], place_pos[1], self.place_offset, 0
        return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)

  def planHouseBuilding3(self, blocks, bricks, roofs):
    valid_block_pos = self.blockPosValidHouseBuilding2(blocks)
    # not holding, do pick
    if not self._isHolding():
      if not valid_block_pos:
        if self._checkOnTop(bricks[0], roofs[0]):
          return self.getPickingRoofPlan(roofs)
        # block pos not valid, and brick on top of any block => pick brick
        elif self._checkOnTop(blocks[0], bricks[0]) or self._checkOnTop(blocks[1], bricks[0]):
          return self.getPickingBrickPlan(bricks)
        # block pos not valid, and roof on top of any block => pick roof
        elif self._checkOnTop(blocks[0], roofs[0]) or self._checkOnTop(blocks[1], roofs[0]):
          return self.getPickingRoofPlan(roofs)
        else:
          # block pos not valid, adjust block pos => pick block
          return self.getPickingBlockPlan(blocks)
      else:
        if not self.brickPosValidHouseBuilding3(blocks, bricks):
          # block pos valid, brick is not on top, roof on top of brick => pick roof
          if self._checkOnTop(bricks[0], roofs[0]):
            return self.getPickingRoofPlan(roofs)
          # block pos valid, brick is not on top, and roof on top of any block => pick roof
          elif self._checkOnTop(blocks[0], roofs[0]) or self._checkOnTop(blocks[1], roofs[0]):
            return self.getPickingRoofPlan(roofs)
          # block pos valid, brick is not on top => pick brick
          else:
            return self.getPickingBrickPlan(bricks)
        # block pos valid, brick is on top => pick roof
        else:
          return self.getPickingRoofPlan(roofs)
    # holding, do placing
    else:
      if self._isObjectHeld(bricks[0]):
        # holding brick, but block pos not valid => place brick on arbitrary pos
        if not valid_block_pos:
          existing_pos = [self._getObjectPosition(o)[:-1] for o in blocks + roofs]
          place_pos = self._getValidPositions(self.max_block_size * 3,
                                              self.max_block_size * 3,
                                              existing_pos,
                                              1)[0]
          x, y, z, r = place_pos[0], place_pos[1], self.place_offset, 0
          return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)
        # holding brick, block pos valid => place brick on top
        else:
          block_pos = [self._getObjectPosition(o) for o in blocks]
          middle_point = np.mean((np.array(block_pos[0]), np.array(block_pos[1])), axis=0)
          x, y, z = middle_point[0], middle_point[1], middle_point[2] + self.place_offset
          slop = (block_pos[0][1] - block_pos[1][1]) / (block_pos[0][0] - block_pos[1][0])
          r = -np.arctan(slop) - np.pi / 2
          while r < 0:
            r += np.pi
          return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)

      elif self._isObjectHeld(roofs[0]):
        # holding roof, but block pos not valid or brick not on top of blocks => place roof on arbitrary pos
        if not (valid_block_pos and self.brickPosValidHouseBuilding3(blocks, bricks)):
          existing_pos = [self._getObjectPosition(o)[:-1] for o in blocks + bricks]
          try:
            place_pos = self._getValidPositions(self.max_block_size * 3,
                                              self.max_block_size * 3,
                                              existing_pos,
                                              1)[0]
          except NoValidPositionException:
            place_pos = self._getValidPositions(self.max_block_size * 3,
                                              self.max_block_size * 3,
                                              [],
                                              1)[0]
          x, y, z, r = place_pos[0], place_pos[1], self.place_offset, 0
          return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)
        # holding roof, block and brick pos valid => place roof on top
        else:
          brick_pos, brick_rot = pb_obj_generation.getObjectPose(bricks[0])
          brick_rot = pb.getEulerFromQuaternion(brick_rot)
          r = -(brick_rot[2] + np.pi / 2)
          while r < 0:
            r += np.pi
          while r > np.pi:
            r -= np.pi
          x, y, z = brick_pos[0], brick_pos[1], brick_pos[2]+self.place_offset
          return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)
      # holding block, place block on valid pos
      else:
        max_block_size = self.max_block_size
        def dist_valid(d):
          return 1.7 * max_block_size < d < 1.8 * max_block_size

        if self._isObjectHeld(blocks[0]):
          other_block = blocks[1]
        else:
          other_block = blocks[0]
        other_block_pos = self._getObjectPosition(other_block)
        roof_pos = [self._getObjectPosition(roofs[0])[:-1]]
        brick_pos = [self._getObjectPosition(bricks[0])[:-1]]
        place_pos = self._getValidPositions(self.max_block_size * 2, self.max_block_size * 2, [], 1)[0]
        for i in range(10000):
          try:
            place_pos = self._getValidPositions(self.max_block_size * 2,
                                                self.max_block_size * 2,
                                                roof_pos+brick_pos,
                                                1)[0]
          except NoValidPositionException:
            continue
          dist = np.linalg.norm(np.array(other_block_pos[:-1]) - np.array(place_pos))
          if dist_valid(dist):
            break
        slop = (place_pos[1] - other_block_pos[1]) / (place_pos[0] - other_block_pos[0])
        r = -np.arctan(slop) - np.pi / 2
        while r < 0:
          r += np.pi
        x, y, z = place_pos[0], place_pos[1], self.place_offset
        return self._encodeAction(self.PLACE_PRIMATIVE, x, y, z, r)

  def planBlockStacking(self):
    # pick
    if not self._isHolding():
      return self.getPickingBlockPlan(self.objects, True)

    # place
    else:
      return self.getStackingBlockPlan(self.objects)

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
    for _ in range(100):
      existing_positions_copy = deepcopy(existing_positions)
      valid_positions = []
      for i in range(num_shapes):
        # Generate random drop config
        x_extents = self.workspace[0][1] - self.workspace[0][0]
        y_extents = self.workspace[1][1] - self.workspace[1][0]

        is_position_valid = False
        for j in range(100):
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
    raise NoValidPositionException

  def _generateShapes(self, shape_type=0, num_shapes=1, scale=None, pos=None, rot=None,
                           min_distance=0.1, padding=0.2, random_orientation=False):
    ''''''
    if shape_type == self.CUBE or shape_type == self.TRIANGLE:
      min_distance = self.block_original_size * self.block_scale_range[1] * 2.4
      padding = self.block_original_size * self.block_scale_range[1] * 2
    elif shape_type == self.BRICK:
      min_distance = self.max_block_size * 4
      padding = self.max_block_size * 3
    elif shape_type == self.ROOF:
      min_distance = self.max_block_size * 4
      padding = self.max_block_size * 3
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
      elif shape_type == self.ROOF:
        handle = pb_obj_generation.generateRoof(position, orientation, scale)
      else:
        raise NotImplementedError
      shape_handles.append(handle)
    self.objects.extend(shape_handles)
    
    for h in shape_handles:
      self.object_types[h] = shape_type

    self.wait(50)
    return shape_handles

  def getObjectPoses(self):
    obj_poses = list()
    for obj in self.objects:
      if self._isObjectHeld(obj):
        continue
      pos, rot = self._getObjectPose(obj)
      obj_poses.append(pos + rot)
    return np.array(obj_poses)

  def _getObjectPose(self, obj):
    return pb_obj_generation.getObjectPose(obj)

  def getObjectPositions(self):
    obj_positions = list()
    for obj in self.objects:
      if self._isObjectHeld(obj):
        continue
      obj_positions.append(self._getObjectPosition(obj))
    return np.array(obj_positions)

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

  def _isObjOnTop(self, obj, objects=None):
    if not objects:
      objects = self.objects
    obj_position = self._getObjectPosition(obj)
    for o in objects:
      if self._isObjectHeld(o) or o is obj:
        continue
      block_position = self._getObjectPosition(o)
      if np.allclose(block_position[:-1], obj_position[:-1],
                     atol=self.block_original_size * self.block_scale_range[0] * 2 / 3) and \
          block_position[-1] > obj_position[-1]:
        return False
    return True

  def _getNumTopBlock(self, blocks=None):
    if not blocks:
      blocks = self.objects
    cluster_pos = []
    for obj in blocks:
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
    elif obj_type is self.TRIANGLE or obj_type is self.ROOF:
      angle = abs(angle - np.pi/2)
      angle = min(angle, np.pi - angle)
    return angle < np.pi / 12

  def _checkPerfectPlace(self, x, y, z, rot, objects):
    end_pos = np.array([x, y, z])
    sorted_obj = sorted(objects, key=lambda o: np.linalg.norm(end_pos - pb_obj_generation.getObjectPosition(o)))
    obj_pos, obj_rot = pb_obj_generation.getObjectPose(sorted_obj[0])
    obj_type = self.object_types[sorted_obj[0]]
    obj_rot = pb.getEulerFromQuaternion(obj_rot)
    angle = np.pi - np.abs(np.abs(rot - obj_rot[2]) - np.pi)
    if angle > np.pi/2:
      angle -= np.pi/2
    angle = min(angle, np.pi / 2 - angle)
    return angle < np.pi / 12


  def _checkObjUpright(self, obj):
    triangle_rot = pb_obj_generation.getObjectRotation(obj)
    triangle_rot = pb.getEulerFromQuaternion(triangle_rot)
    return abs(triangle_rot[0]) < np.pi/9 and abs(triangle_rot[1]) < np.pi/9

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
      if p[2] == bottom_obj:
        return True
    return False

  def _checkInBetween(self, obj0, obj1, obj2, threshold=None):
    if not threshold:
      threshold = self.max_block_size
    position0 = pb_obj_generation.getObjectPosition(obj0)[:-1]
    position1 = pb_obj_generation.getObjectPosition(obj1)[:-1]
    position2 = pb_obj_generation.getObjectPosition(obj2)[:-1]
    middle_point = np.mean((np.array(position1), np.array(position2)), axis=0)
    dist = np.linalg.norm(middle_point - position0)
    return dist < threshold

  def _checkOriSimilar(self, objects, threshold=np.pi/7):
    oris = list(map(lambda o: pb.getEulerFromQuaternion(pb_obj_generation.getObjectRotation(o))[2], objects))
    return np.allclose(oris, oris, threshold)

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
    if self._isHolding():
      if self.object_types[self.robot.holding_obj] is self.BRICK:
        extend = int(2*self.max_block_size/self.heightmap_resolution)
      elif self.object_types[self.robot.holding_obj] is self.ROOF:
        extend = int(1.5*self.max_block_size/self.heightmap_resolution)
      else:
        extend = int(0.5*self.max_block_size/self.heightmap_resolution)
    else:
      extend = int(0.5*self.max_block_size/self.heightmap_resolution)
    local_region = self.heightmap[int(max(y_pixel - extend, 0)):int(min(y_pixel + extend, self.heightmap_size)), \
                                  int(max(x_pixel - extend, 0)):int(min(x_pixel + extend, self.heightmap_size))]
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
