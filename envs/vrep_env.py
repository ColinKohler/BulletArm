import sys
import time
import numpy as np
import numpy.random as npr
import scipy.misc

from simulation import vrep
from grippers.rg2 import RG2
from robots.ur5 import UR5
from sensors.vision_sensor import VisionSensor
from utils import vrep_utils
from utils import transformations

# Motion primatives
PICK_PRIMATIVE = 0
PLACE_PRIMATIVE = 1
PUSH_PRIMATIVE = 2

class VrepEnv(object):
  '''
  RL environment setup in a similar manner to OpenAI Gym envs (step, reset).
  This is the base VRep env which should be extended for different tasks.
  Args:
    - seed: Random seed to use for this environment.
    - max_steps: Maximum number of steps in an episode
    - vrep_ip: IP address of machine where VRep simulator is running
    - vrep_port: Port to communicate with VRep simulator over
    - fast_mode: Teleport the arm when it doesn't interact with other objects.
  '''
  def __init__(self, seed, workspace, max_steps=10, heightmap_size=250,
                     vrep_ip='127.0.0.1', vrep_port=19997, fast_mode=False):
    # Setup workspace
    self.workspace = workspace
    self.workspace_size = np.linalg.norm(self.workspace[0,1] - self.workspace[0,0])
    self.max_steps = max_steps
    self.fast_mode = fast_mode

    # Setup depth image parameters
    self.cam_intrinsics = np.asarray([[618.62, 0, 320], [0, 618.62, 240], [0, 0, 1]])
    self.heightmap_size = heightmap_size
    self.heightmap_shape = (self.heightmap_size, self.heightmap_size, 1)
    self.heightmap_resolution = self.workspace_size / self.heightmap_size

    # VRep simulator
    self.vrep_ip = vrep_ip
    self.vrep_port = vrep_port

    # Setup observation and action spaces
    self.obs_shape = self.heightmap_shape
    self.action_space = np.concatenate((self.workspace[:2,:], np.array([[0.0], [2*np.pi]])), axis=1)
    self.action_shape = 3

    # Set default poses
    self.home_pose = transformations.euler_matrix(np.radians(90), 0, np.radians(90))
    self.home_pose[:2,-1] = (self.workspace[:2,1] + self.workspace[:2,0]) / 2.0
    self.home_pose[2,-1] = self.workspace[2,1] - 0.05

    # Set random numpy seed
    npr.seed(seed)

  def connectToVrep(self):
    '''
    Connect to VRep simulator. Do not want to call this on reset.
    '''
    # Set v-rep parameters
    self.sim_client = vrep_utils.connectToSimulation(self.vrep_ip, self.vrep_port)
    gripper = RG2(self.sim_client)
    self.ur5 = UR5(self.sim_client, gripper)
    self.sensor = VisionSensor(self.sim_client, 'Vision_sensor_persp', self.workspace, self.cam_intrinsics)

  def disconnectToVrep(self):
    '''
    Disconnect from VRep simulator.
    '''
    vrep_utils.stopSimulation(self.sim_client)
    vrep_utils.disconnectToSimulation(self.sim_client)

  def isSimValid(self):
    '''
    Check if VRep simulation is stable by checking if the gripper and objects are within the workspace

    Returns: True if sim is valid, False otherwise
    '''
    for object_handle in self.object_handles:
      sim_ret, object_position = vrep_utils.getObjectPosition(self.sim_client, object_handle)
      if not self._isPointInWorkspace(object_position): return False

    sim_ret, gripper_position = self.ur5.gripper.getPosition()
    return self._isPointInWorkspace(gripper_position)

  def reset(self):
    '''
    Reset the simulation to initial state
    '''
    vrep_utils.restartSimulation(self.sim_client)
    self.ur5.moveTo(self.home_pose, single_step=self.fast_mode)

    self.current_episode_steps = 1
    self.height_map = None
    self.is_holding_object = False
    self.object_handles = list()

    return self._getObservation()

  # Execute the action and step the simulation
  def step(self, action):
    '''
    Take a action in the environment and run the simulation as specified.
    Args:
      - action: The action to take in the environment
    Returns: (obs, reward, done)
      - obs: Observation tuple (depth_img, robot_state)
      - reward: Reward acheived at current timestep
      - done: Boolean flag indicating if the episode is done
    '''
    motion_primative, x, y, rot = action
    T = transformations.euler_matrix(np.radians(90), rot, np.radians(90))
    T[:2,3] = [x, y]
    T[2,3] = self._getPrimativeHeight(motion_primative, x, y)

    #  Execute action
    if motion_primative == PICK_PRIMATIVE:
      self.is_holding_object = self.ur5.pick(T, 0.05 if self.fast_mode else 0.25, fast_mode=self.fast_mode)
    elif motion_primative == PLACE_PRIMATIVE:
      self.ur5.place(T, 0.25, fast_mode=self.fast_mode)
      self.is_holding_object = False

    # Move to home position
    if self.is_holding_object:
      self.ur5.moveTo(self.home_pose)
    else:
      self.ur5.moveTo(self.home_pose, single_step=self.fast_mode)

    # Check for termination and get reward
    obs = self._getObservation()
    done = self._checkTermination()
    reward = 1.0 if done else 0.0

    # Check to see if we are at the max steps
    if not done:
      done = self.current_episode_steps >= self.max_steps
    self.current_episode_steps += 1

    # Check to see if sim is valid and reset otherwise
    if not done and not self.isSimValid():
      done = True

    return obs, reward, done

  def _getObservation(self):
    '''
    Helper method to get the observation tuple
    Returns: Observation tuple - (is_holding_object, depth_img)
      - is_holding_object: boolean indicating if the robot is currently holdinga object
      - depth_img: Depth image of workspace (DxDx1 numpy array)
    '''
    data = self.sensor.getData()
    depth_heightmap, rgb_heightmap = self.sensor.getHeightmap()

    depth_heightmap = scipy.misc.imresize(depth_heightmap, (self.heightmap_size, self.heightmap_size), mode='F')
    self.heightmap = depth_heightmap

    return (self.is_holding_object, depth_heightmap.reshape([self.heightmap_size, self.heightmap_size, 1]))

  def _checkTermination(self):
    '''
    Sub-envs should override this to set their own termination conditions
    Returns: False
    '''
    return False

  def _getPrimativeHeight(self, motion_primative, x, y):
    '''
    Get the z position for the given action using the current heightmap.
    Args:
      - motion_primative: Pick/place motion primative
      - x: X coordinate for action
      - y: Y coordinate for action
    Returns: Valid Z coordinate for the action
    '''
    x_pixel, y_pixel = self._getPixelsFromPos(x, y)
    local_region = self.heightmap[max(y_pixel - 30, 0):min(y_pixel + 30, 250), \
                                  max(x_pixel - 30, 0):min(x_pixel + 30, 250)]
    safe_z_pos = np.max(local_region) + self.workspace[2][0]
    safe_z_pos = safe_z_pos - 0.01 if motion_primative == PICK_PRIMATIVE else safe_z_pos + 0.01

    return safe_z_pos

  def _getPixelsFromPos(self, x, y):
    '''
    Get the x/y pixels on the heightmap for the given coordinates
    Args:
      - x: X coordinate
      - y: Y coordinate
    Returns: (x, y) in pixels corresponding to coordinates
    '''
    x_pixel = (x - self.workspace[0][0]) / self.heightmap_resolution
    y_pixel = (y - self.workspace[1][0]) / self.heightmap_resolution

    return int(x_pixel), int(y_pixel)

  # TODO: Fix this up.
  def _generateShapes(self, shape_type, num_shapes, size=None, min_distance=0.1, padding=0.2, sleep_time=0.5):
    '''
    Generate shapes at random positions in the workspace.
    Args:
      - shape_type: Type of shape to be generate (0: cuboid, 1: sphere, 2: cylinder, 3: cone) - num_shapes: Number of shapes to generate
      - size: Specify the desired size of shapes to generate, otherwise random
    Returns: True if shape generation succeeded, False otherwise
    '''
    shape_handles = list()
    positions = list()

    shape_name = self._getShapeName(shape_type)
    for i in range(num_shapes):
      name = '{}_{}'.format(shape_name, len(shape_handles))
      color = [255.0, 0.0, 0.0]
      size = size if size else list(np.concatenate((npr.uniform(low=0.035, high=0.045, size=2), [0.03]), axis=0))
      mass = 0.1

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

      # orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
      orientation = [0., 0., 0.]

      handle = vrep_utils.generateShape(self.sim_client, name, shape_type, size, position, orientation, mass, color)
      if handle is None:
        return False, None
      shape_handles.append(handle)
      time.sleep(sleep_time )

    self.object_handles.extend(shape_handles)
    return True, shape_handles

  def _getShapeName(self, shape_type):
    ''' Get the shape name from the type (int) '''
    if shape_type == 0: return 'cube'
    elif shape_type == 1: return 'sphere'
    elif shape_type == 2: return 'cylinder'
    elif shape_type == 3: return 'cone'
    else: return 'unknown'

  def _isPointInWorkspace(self, p):
    '''
    Checks if the given point is within the workspace

    Args:
      - p: [x, y, z] point

    Returns: True in point is within workspace, False otherwise
    '''
    return p[0] > self.workspace[0][0] - 0.1 and p[0] < self.workspace[0][1] + 0.1 and \
           p[1] > self.workspace[1][0] - 0.1 and p[1] < self.workspace[1][1] + 0.1 and \
           p[2] > self.workspace[2][0] and p[2] < self.workspace[2][1]
