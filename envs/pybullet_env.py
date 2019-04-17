import time
from copy import deepcopy
import numpy as np
import numpy.random as npr

import pybullet as pb
import pybullet_data

from helping_hands_rl_envs.envs.base_env import BaseEnv
from helping_hands_rl_envs.pybullet_toolkit.robots.ur5_rg2 import UR5_RG2
import helping_hands_rl_envs.pybullet_toolkit.utils.object_generation as pb_obj_generation

class PyBulletEnv(BaseEnv):
  '''
  PyBullet Arm RL base class.
  '''
  def __init__(self, seed, workspace, max_steps=10, heightmap_size=250, fast_mode=False, render=False, action_sequence='pxyr'):
    super(PyBulletEnv, self).__init__(seed, workspace, max_steps, heightmap_size, action_sequence)

    # Connect to pybullet and add data files to path
    if render:
      self.client = pb.connect(pb.GUI)
    else:
      self.client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    self.dynamic = not fast_mode

    # Environment specific variables
    self._timestep = 1. / 240.
    self.ur5 = UR5_RG2()
    self.pick_offset = 0.25
    self.place_offset = 0.25
    self.block_original_size = 0.05
    self.block_scale_range = (0.5, 0.7)

    # Setup camera parameters
    self.view_matrix = pb.computeViewMatrixFromYawPitchRoll([0.5, 0.0, 0], 1.0, -90, -90, 0, 2)
    self.proj_matrix = pb.computeProjectionMatrix(-0.25, 0.25, -0.25, 0.25, -1.0, 10.0)

    # Rest pose for arm
    rot = pb.getQuaternionFromEuler([0,np.pi,0])
    self.rest_pose = [[0.0, 0.5, 0.5], rot]

  def reset(self):
    ''''''
    pb.resetSimulation()
    pb.setTimeStep(self._timestep)

    pb.setGravity(0, 0, -10)
    self.table_id = pb.loadURDF('plane.urdf', [0,0,0])

    # Load the UR5 and set it to the home positions
    self.ur5.reset()

    # Reset episode vars
    self.objects = list()
    self.heightmap = None
    self.current_episode_steps = 1

    # Step simulation
    pb.stepSimulation()

    return self._getObservation()

  def saveState(self):
    self.state = {'current_episode_steps': deepcopy(self.current_episode_steps),
                  'objects': deepcopy(self.objects),
                  'env_state': pb.saveState()}

  def restoreState(self):
    self.current_episode_steps = self.state['current_episode_steps']
    self.objects = self.state['objects']
    pb.restoreState(self.state['env_state'])

  def step(self, action):
    ''''''
    motion_primative, x, y, z, rot = self._getSpecificAction(action)

    # Get transform for action
    pos = [x, y, z]
    rot = pb.getQuaternionFromEuler([0, np.pi, rot])

    # Take action specfied by motion primative
    if motion_primative == self.PICK_PRIMATIVE:
      self.ur5.pick(pos, rot, self.pick_offset, dynamic=self.dynamic)
    elif motion_primative == self.PLACE_PRIMATIVE:
      if self.ur5.is_holding:
        self.ur5.place(pos, rot, self.place_offset, dynamic=self.dynamic)
    elif motion_primative == self.PUSH_PRIMATIVE:
      pass
    else:
      raise ValueError('Bad motion primative supplied for action.')

    if self.ur5.is_holding:
      self.ur5.moveTo(self.rest_pose[0], self.rest_pose[1], dynamic=True)
    else:
      self.ur5.moveTo(self.rest_pose[0], self.rest_pose[1], dynamic=self.dynamic)

    for _ in range(100):
      pb.stepSimulation()

    # Check for termination and get reward
    obs = self._getObservation()
    done = self._checkTermination()
    reward = 1.0 if done else 0.0

    # Check to see if we are at the max steps
    if not done:
      done = self.current_episode_steps >= self.max_steps
    self.current_episode_steps += 1

    return obs, reward, done

  def _getObservation(self):
    ''''''
    image_arr = pb.getCameraImage(width=self.heightmap_size, height=self.heightmap_size,
                                  viewMatrix=self.view_matrix, projectionMatrix=self.proj_matrix)
    self.heightmap = image_arr[3] - np.min(image_arr[3])
    self.heightmap = self.heightmap.T

    return self.getHoldingState(), self.heightmap.reshape([self.heightmap_size, self.heightmap_size, 1])

  def _generateShapes(self, shape_type, num_shapes, size=None, pos=None, rot=None,
                           min_distance=0.1, padding=0.2, random_orientation=False):
    ''''''
    shape_handles = list()
    positions = list()

    shape_name = self._getShapeName(shape_type)
    for i in range(num_shapes):
      name = '{}_{}'.format(shape_name, len(shape_handles))

      # Generate random drop config
      x_extents = self.workspace[0][1] - self.workspace[0][0]
      y_extents = self.workspace[1][1] - self.workspace[1][0]

      is_position_valid = False
      while not is_position_valid:
        position = [(x_extents - padding) * npr.random_sample() + self.workspace[0][0] + padding / 2,
                    (y_extents - padding) * npr.random_sample() + self.workspace[1][0] + padding / 2,
                    0.05]
        if positions:
          is_position_valid = np.all(np.sum(np.abs(np.array(positions) - np.array(position[:-1])), axis=1) > min_distance)
        else:
          is_position_valid = True
      positions.append(position[:-1])
      if random_orientation:
        orientation = pb.getQuaternionFromEuler([0., 0., 2*np.pi*np.random.random_sample()])
      else:
        orientation = pb.getQuaternionFromEuler([0., 0., 0.])

      scale = npr.uniform(self.block_scale_range[0], self.block_scale_range[1])

      handle = pb_obj_generation.generateCube(position, orientation, scale)
      shape_handles.append(handle)

    self.objects.extend(shape_handles)
    for _ in range(50):
      pb.stepSimulation()
    return shape_handles

  def _getObjectPosition(self, obj):
    return pb_obj_generation.getObjectPosition(obj)

  def getHoldingState(self):
    return self.ur5.is_holding

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
      self.ur5.openGripper()
      self.objects.remove(obj)

  def _moveObjectOutWorkspace(self, obj):
    pos = [-0.50, 0, 0.25]
    pb.resetBasePositionAndOrientation(obj, pos, pb.getQuaternionFromEuler([0., 0., 0.]))

  def _getNumTopBlock(self):
    cluster_pos = []
    for obj in self.objects:
      if self._isObjectHeld(obj):
        return -1
      block_position = self._getObjectPosition(obj)
      cluster_flag = False
      for cluster in cluster_pos:
        if np.allclose(block_position[:-1], cluster, atol=self.block_original_size*self.block_scale_range[0]/2):
          cluster.append(block_position[:-1])
          cluster_flag = True
          break
      if not cluster_flag:
        cluster_pos.append([block_position[:-1]])
    return len(cluster_pos)
