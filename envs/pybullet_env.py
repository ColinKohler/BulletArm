import time
import numpy as np
import numpy.random as npr

import pybullet as pb
import pybullet_data

from base_env import BaseEnv

class PyBulletEnv(BaseEnv):
  '''
  PyBullet Arm RL base class.
  '''
  def __init__(self, seed, workspace, max_steps=10, heightmap_size=250, fast_mode=False):
    super(VrepEnv, self).__init__(seed, workspace, max_steps, heightmap_size)

    # Connect to pybullet and add data files to path
    self.client = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Environment specific variables
    self._timestep = 1. / 240.

    # Setup camera parameters
    self.view_matrix = pb.computeViewMatrixFromYawPitchRoll([0.5, 0.0, 0], 1.0, 180, 0, 0, 1)
    self.proj_matrix = pb.computeProjectionMatrix(-0.25, 0.25, -0.25, 0.25, -1.0, 10.0)

  def reset(self):
    ''''''
    pb.resetSimulation()
    pb.setTimeStep(self._timestep)

    pb.setGravity(0, 0, -10)
    self.table_id = pb.loadURDF('plane.urdf', [0,0,0])

    # Load the UR5 and set it to the home positions

    # Step simulation
    pb.stepSimulation()

    return self._getObservation()

  def step(self, action):
    ''''''
    motion_primative, x, y, rot = action

    # Get transfor for action
    T = transformations.euler_matrix(np.radians(90), rot, np.radians(90))
    T[:2,3] = [x, y]
    T[2,3] = self._getPrimativeHeight(motion_primative, x, y)

    Take action specfied by motion primative
    if motion_primative == PICK_PRIMATIVE:
      pass
    elif motion_primative == PLACE_PRIMATIVE:
      pass
    elif motion_primative == PUSH_PRIMATIVE:
      pass
    else:
      raise ValueError('Bad motion primative supplied for action.')

    # Step simulation
    pb.stepSimulation()

    return self._getState()

  def _getObservation(self):
    ''''''
    image_arr = pb.getCameraImage(width=self.heightmap_size, height=self.heightmap_size,
                                  viewMatrix=self.view_matrix, projectionMatrix=self.proj_matrix)
    self.heightmap = image_arr[3]

    return False, self.heightmap

  def generateShapes(self, shape_type, size=None, pos=None, rot=None):
    '''
    Generate shapes at random positions in the workspace.
    Args:
      - shape_type: Type of shape to be generate (0: cuboid, 1: sphere, 2: cylinder, 3: cone) - num_shapes: Number of shapes to generate
      - size: Specify the desired size of shapes to generate, otherwise random
    Returns: True if shape generation succeeded, False otherwise
    '''
    pass
