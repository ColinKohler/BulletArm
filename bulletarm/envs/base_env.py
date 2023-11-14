'''
.. moduleauthor: Colin Kohler <github.com/ColinKohler>
'''

import os
import pickle
import copy
import numpy as np
import numpy.random as npr
from scipy.ndimage import median_filter
import skimage.transform as sk_transform

import pybullet as pb
import pybullet_data

import bulletarm.pybullet.utils.constants as constants
from bulletarm.pybullet.utils import transformations
import bulletarm.envs.configs as env_configs

from bulletarm.pybullet.robots.ur5_simple import UR5_Simple
from bulletarm.pybullet.robots.ur5_robotiq import UR5_Robotiq
from bulletarm.pybullet.robots.kuka import Kuka
from bulletarm.pybullet.robots.panda import Panda
from bulletarm.pybullet.utils.sensor import Sensor
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
import bulletarm.pybullet.utils.object_generation as pb_obj_generation
from bulletarm.pybullet.utils.constants import NoValidPositionException

class BaseEnv:
  '''
  Base Env Class.

  Args:
    config: Config used to specify various environment details
  '''
  def __init__(self, config):
    # Load the default config and replace any duplicate values with the config
    config = {**env_configs.DEFAULT_CONFIG, **config}

    self.seed = config['seed']
    npr.seed(self.seed)

    # Setup environment
    self.workspace = config['workspace']
    self.workspace_size = np.linalg.norm(self.workspace[0,1] - self.workspace[0,0])
    self.pos_candidate = config['pos_candidate'].astype(np.int) if config['pos_candidate'] else None
    self.max_steps = config['max_steps']

    # Setup heightmap
    self.heightmap_size = config['obs_size']
    self.in_hand_size = config['in_hand_size']
    self.in_hand_mode = config['in_hand_mode']
    self.heightmap_shape = (self.heightmap_size, self.heightmap_size, 1)
    self.heightmap_resolution = self.workspace_size / self.heightmap_size

    # Setup action format
    assert config['action_sequence'].find('x') != -1
    assert config['action_sequence'].find('y') != -1
    self.action_sequence = config['action_sequence']

    # Setup observation and action spaces
    self.obs_shape = self.heightmap_shape
    self.num_primatives = constants.NUM_PRIMATIVES

    self.action_space = [[], []]
    primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a), ['p', 'x', 'y', 'z', 'r'])
    if primative_idx != -1:
      self.action_has_primative = True
    if x_idx != -1:
      self.action_space[0].append(self.workspace[0,0])
      self.action_space[1].append(self.workspace[0,1])
    if y_idx != -1:
      self.action_space[0].append(self.workspace[1,0])
      self.action_space[1].append(self.workspace[1,1])
    if z_idx != -1:
      self.action_space[0].append(self.workspace[2,0])
      self.action_space[1].append(self.workspace[2,1])
    if rot_idx != -1:
      self.action_space[0].append(0.0)
      self.action_space[1].append(np.pi)

    self.action_space = np.array(self.action_space)
    self.action_shape = self.action_space.shape[0]

    # Connect to pybullet and add data files to path
    if config['render']:
      self.client = pb.connect(pb.GUI)
      # For screenshotting envs
      # pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)
      # pb.resetDebugVisualizerCamera(
      #     cameraDistance=1.1,
      #     cameraYaw=90,
      #     cameraPitch=-40,
      #     cameraTargetPosition=[0, 0, 0])
    else:
      self.client = pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Environment specific variables
    self.dynamic = not config['fast_mode']
    self._timestep = 1. / 240.

    # Setup robot
    if config['robot'] == 'ur5':
      self.robot = UR5_Simple()
    elif config['robot'] == 'ur5_robotiq':
      self.robot = UR5_Robotiq()
    elif config['robot'] == 'kuka':
      self.robot = Kuka()
    elif config['robot'] == 'panda':
      self.robot = Panda()
    else:
      raise NotImplementedError

    # Setup physics mode
    if config['physics_mode'] == 'fast':
      self.physics_mode = 'fast'
      self.num_solver_iterations = 50
      self.solver_residual_threshold = 1e-7
      self.robot.position_gain = 0.02
    elif config['physics_mode'] == 'slow':
      self.physics_mode = 'slow'
      self.num_solver_iterations = 200
      self.solver_residual_threshold = 1e-7
      self.robot.position_gain = 0.01
    elif config['physics_mode'] == 'custom':
      self.physics_mode = 'custom'
      self.num_solver_iterations = config['num_solver_iterations']
      self.solver_residual_threshold = config['solver_residual_threshold']
      self.robot.position_gain = 0.01

    # TODO: Move this somewhere it makes sense
    self.block_original_size = 0.05
    self.block_scale_range = config['object_scale_range']
    self.min_block_size = self.block_original_size * self.block_scale_range[0]
    self.max_block_size = self.block_original_size * self.block_scale_range[1]

    self.pick_pre_offset = 0.10
    self.pick_offset = self.block_scale_range[1]*self.block_original_size/2
    self.place_pre_offset = 0.10
    self.place_offset = self.block_scale_range[1]*self.block_original_size/2
    self.pull_offset = 0.25

    self.object_init_z = 0.020

    # Setup camera
    ws_size = max(self.workspace[0][1] - self.workspace[0][0], self.workspace[1][1] - self.workspace[1][0])
    cam_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 10]
    target_pos = [self.workspace[0].mean(), self.workspace[1].mean(), 0]
    cam_up_vector = [-1, 0, 0]
    self.sensor = Sensor(cam_pos, cam_up_vector, target_pos, ws_size, cam_pos[2] - 1, cam_pos[2])

    # Rest pose for arm
    rot = pb.getQuaternionFromEuler([0, np.pi, 0])
    self.rest_pose = [[0.0, 0.5, 0.5], rot]

    self.objects = list()
    self.object_types = {}

    self.simulate_grasp = config['simulate_grasp']
    self.perfect_grasp = config['perfect_grasp']
    self.perfect_place = config['perfect_place']
    self.workspace_check = config['workspace_check']
    self.num_random_objects = config['num_random_objects']
    self.random_orientation = config['random_orientation']
    self.check_random_obj_valid = config['check_random_obj_valid']
    self.random_orientation = config['random_orientation']
    self.num_obj = config['num_objects']
    self.reward_type = config['reward_type']
    self.object_type = config['object_type']
    self.hard_reset_freq = config['hard_reset_freq']
    self.min_object_distance = config['min_object_distance']
    self.min_boarder_padding = config['min_boarder_padding']
    self.deconstruct_init_offset = config['deconstruct_init_offset']
    self.pick_top_down_approach = config['pick_top_down_approach']
    self.place_top_down_approach = config['place_top_down_approach']
    self.half_rotation = config['half_rotation']
    self.white_plane = 'white_plane' in config['workspace_option']
    self.black_workspace = 'black_workspace' in config['workspace_option']
    self.trans_plane = 'trans_plane' in config['workspace_option']
    self.trans_robot = 'trans_robot' in config['workspace_option']
    if self.trans_robot and config['robot'] != 'kuka':
      raise NotImplementedError

    self.robot.adjust_gripper_after_lift = config['adjust_gripper_after_lift']
    if config['robot'] == 'kuka':
      self.robot.adjust_gripper_offset = config['kuka_adjust_gripper_offset']

    self.episode_count = -1
    self.table_id = None
    self.heightmap = None
    self.current_episode_steps = 1
    self.last_action = None
    self.last_obj = None
    self.state = {}
    self.pb_state = None

  def initialize(self):
    '''
    Initialize the pybullet world.
    '''
    pb.resetSimulation()
    pb.setPhysicsEngineParameter(numSubSteps=0,
                                 numSolverIterations=self.num_solver_iterations,
                                 solverResidualThreshold=self.solver_residual_threshold,
                                 constraintSolverType=pb.CONSTRAINT_SOLVER_LCP_SI)
    pb.setTimeStep(self._timestep)
    pb.setGravity(0, 0, -10)

    # TODO: These might have to be in the config depending on how they effect the solver_residual_threshold
    if self.white_plane:
      self.table_id = pb.loadURDF(os.path.join(constants.URDF_PATH, 'white_plane.urdf'), [0,0,0])
    else:
      self.table_id = pb.loadURDF('plane.urdf', [0, 0, 0])
    if self.trans_plane:
      pb.changeVisualShape(self.table_id, -1, rgbaColor=[0, 0, 0, 0])
    pb.changeDynamics(self.table_id, -1, linearDamping=0.04, angularDamping=0.04, restitution=0, contactStiffness=3000, contactDamping=100)

    if self.black_workspace:
      ws_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[self.workspace_size/2, self.workspace_size/2, 0.001], rgbaColor=[0.2, 0.2, 0.2, 1])
      ws_id = pb.createMultiBody(baseMass=0,
                                 baseVisualShapeIndex=ws_visual,
                                 basePosition=[self.workspace[0].mean(), self.workspace[1].mean(), 0],
                                 baseOrientation=[0, 0, 0, 1])

    # Load the UR5 and set it to the home positions
    self.robot.initialize()
    if self.trans_robot:
      for i in range(-1, 9):
        pb.changeVisualShape(self.robot.id, i, rgbaColor=[1, 1, 1, 0])
      pb.changeVisualShape(self.robot.id, 11, rgbaColor=[1, 1, 1, 0])
    # Reset episode vars
    self.objects = list()
    self.object_types = {}

    self.heightmap = None
    self.current_episode_steps = 1
    self.last_action = None

    # Step simulation
    pb.stepSimulation()

  def resetPybulletWorkspace(self):
    '''
    Reset the PyBullet world to just contain the robot.
    '''
    # soft reset has bug in older pybullet versions. 2.7,1 works good
    self.episode_count += 1
    if self.episode_count % self.hard_reset_freq == 0:
      self.initialize()
      self.episode_count = 0
    else:
      for o in self.objects:
        pb.removeBody(o.object_id)
      self.robot.reset()
      self.objects = list()
      self.object_types = {}
      self.heightmap = None
      self.current_episode_steps = 1
      self.last_action = None
      self.last_obj = None
      self.state = {}
      self.pb_state = None

    while True:
      try:
        self._generateShapes(constants.RANDOM, self.num_random_objects, random_orientation=True)
      except Exception as e:
        continue
      else:
        break

    pb.stepSimulation()

  def reset(self):
    '''
    Reset the environment.

    Returns: Numpy vector of the observation
    '''
    self.resetPybulletWorkspace()
    return self._getObservation()

  def step(self, action):
    '''
    Take an action in the enviornment.

    Args:
      - action: Int indicating the action to take

    Returns: (obs, reward, done)
      - obs: Numpy vector of the observation
      - reward: Double of the reward given
      - done: Bool flag indicating if the episode is done
    '''
    self.takeAction(action)
    self.wait(100)
    obs = self._getObservation(action)
    done = self._checkTermination()
    reward = 1.0 if done else 0.0

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def takeAction(self, action):
    motion_primative, x, y, z, rot = self._decodeAction(action)
    self.last_action = action
    self.last_obj = self.robot.holding_obj

    # Get transform for action
    pos = [x, y, z]
    rot_q = pb.getQuaternionFromEuler(rot)

    # Take action specfied by motion primative
    if motion_primative == constants.PICK_PRIMATIVE:
      if self.robot.holding_obj is None:
        if self.perfect_grasp and not self._checkPerfectGrasp(x, y, z, rot, self.objects):
          return
        self.robot.pick(pos, rot_q, self.pick_pre_offset, dynamic=self.dynamic,
                        objects=self.objects, simulate_grasp=self.simulate_grasp, top_down_approach=self.pick_top_down_approach)
    elif motion_primative == constants.PLACE_PRIMATIVE:
      obj = self.robot.holding_obj
      if self.robot.holding_obj is not None:
        if self.perfect_place and not self._checkPerfectPlace(x, y, z, rot, self.objects):
          return
        self.robot.place(pos, rot_q, self.place_pre_offset,
                         dynamic=self.dynamic, simulate_place=self.simulate_grasp, top_down_approach=self.place_top_down_approach)
    elif motion_primative == constants.PUSH_PRIMATIVE:
      pass
    elif motion_primative == constants.PULL_PRIMATIVE:
      self.robot.pull(pos, rot_q, self.pull_offset, dynamic=self.dynamic)
    else:
      raise ValueError('Bad motion primative supplied for action.')

  def isSimValid(self):
    for obj in self.objects:
      p = obj.getPosition()
      if not self.check_random_obj_valid and self.object_types[obj] == constants.RANDOM:
        continue
      if self._isObjectHeld(obj):
        continue
      if self.workspace_check == 'point':
        if not self._isPointInWorkspace(p):
          return False
      else:
        if not self._isObjectWithinWorkspace(obj):
          return False
      if self.pos_candidate is not None:
        if np.abs(self.pos_candidate[0] - p[0]).min() > 0.02 or np.abs(self.pos_candidate[1] - p[1]).min() > 0.02:
          return False
    return True

  def areObjectsInWorkspace(self):
    for obj in self.objects:
      p = obj.getPosition()
      if self._isObjectHeld(obj):
        continue
      if self.workspace_check == 'point':
        if not self._isPointInWorkspace(p):
          return False
      else:
        if not self._isObjectWithinWorkspace(obj):
          return False
    return True

  def wait(self, iteration):
    [pb.stepSimulation() for _ in range(iteration)]

  def didBlockFall(self):
    if self.last_action is None:
      return False

    motion_primative, x, y, z, rot = self._decodeAction(self.last_action)
    obj = self.last_obj

    if obj:
      return motion_primative == constants.PLACE_PRIMATIVE and \
             np.linalg.norm(np.array(obj.getXYPosition()) - np.array([x,y])) > obj.size / 2.
    else:
      return False

  def getValidSpace(self):
    return self.workspace

  def _getObservation(self, action=None):
    ''''''
    old_heightmap = copy.copy(self.heightmap)
    self.heightmap = self._getHeightmap()

    if action is None or self._isHolding() == False:
      in_hand_img = self.getEmptyInHand()
    else:
      motion_primative, x, y, z, rot = self._decodeAction(action)
      in_hand_img = self.getInHandImage(old_heightmap, x, y, z, rot, self.heightmap)

    return self._isHolding(), in_hand_img, self.heightmap.reshape([1, self.heightmap_size, self.heightmap_size])

  def _getHeightmap(self):
    return self.sensor.getHeightmap(self.heightmap_size)

  def _getValidPositions(self, border_padding, min_distance, existing_positions, num_shapes, sample_range=None):
    existing_positions_copy = copy.deepcopy(existing_positions)
    sample_range = copy.deepcopy(sample_range)
    valid_positions = list()
    for i in range(num_shapes):
      # Generate random drop config
      x_extents = self.getValidSpace()[0][1] - self.getValidSpace()[0][0]
      y_extents = self.getValidSpace()[1][1] - self.getValidSpace()[1][0]

      is_position_valid = False
      for j in range(100):
        if is_position_valid:
          break
        if sample_range:
          sample_range[0][0] = max(sample_range[0][0], self.getValidSpace()[0][0]+border_padding/2)
          sample_range[0][1] = min(sample_range[0][1], self.getValidSpace()[0][1]-border_padding/2)
          sample_range[1][0] = max(sample_range[1][0], self.getValidSpace()[1][0]+border_padding/2)
          sample_range[1][1] = min(sample_range[1][1], self.getValidSpace()[1][1]-border_padding/2)
          position = [(sample_range[0][1] - sample_range[0][0]) * npr.random_sample() + sample_range[0][0],
                      (sample_range[1][1] - sample_range[1][0]) * npr.random_sample() + sample_range[1][0]]
        else:
          position = [(x_extents - border_padding) * npr.random_sample() + self.getValidSpace()[0][0] + border_padding / 2,
                      (y_extents - border_padding) * npr.random_sample() + self.getValidSpace()[1][0] +  border_padding / 2]

        if self.pos_candidate is not None:
          position[0] = self.pos_candidate[0][np.abs(self.pos_candidate[0] - position[0]).argmin()]
          position[1] = self.pos_candidate[1][np.abs(self.pos_candidate[1] - position[1]).argmin()]
          if not (self.getValidSpace()[0][0]+border_padding/2 < position[0] < self.getValidSpace()[0][1]-border_padding/2 and
                  self.getValidSpace()[1][0]+border_padding/2 < position[1] < self.getValidSpace()[1][1]-border_padding/2):
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
    else:
      raise NoValidPositionException

  def _getValidOrientation(self, random_orientation):
    if random_orientation:
      orientation = pb.getQuaternionFromEuler([0., 0., 2 * np.pi * np.random.random_sample()])
    else:
      orientation = pb.getQuaternionFromEuler([0., 0., 0.])
    return orientation

  def _getDefaultBoarderPadding(self, shape_type):
    if shape_type in (constants.CUBE, constants.TRIANGLE, constants.RANDOM, constants.CYLINDER, constants.RANDOM_BLOCK,
                      constants.RANDOM_HOUSEHOLD, constants.BOTTLE, constants.TEST_TUBE, constants.SWAB):
      padding = self.max_block_size * 2.4
    elif shape_type in (constants.BRICK, constants.ROOF, constants.CUP, constants.SPOON, constants.BOX,
                        constants.FLAT_BLOCK):
      padding = self.max_block_size * 3.4
    elif shape_type == constants.BOWL:
      padding = 0.17
    elif shape_type == constants.PLATE:
      padding = 0.2
    else:
      raise ValueError('Attempted to generate invalid shape.')
    return padding

  def _getDefaultMinDistance(self, shape_type):
    if shape_type in (constants.CUBE, constants.TRIANGLE, constants.RANDOM, constants.CYLINDER, constants.RANDOM_BLOCK,
                      constants.BOTTLE, constants.TEST_TUBE, constants.SWAB):
      min_distance = self.max_block_size * 2.4
    elif shape_type in (constants.BRICK, constants.ROOF, constants.CUP, constants.SPOON, constants.BOX,
                        constants.FLAT_BLOCK):
      min_distance = self.max_block_size * 3.4
    elif shape_type in [constants.RANDOM_HOUSEHOLD]:
      min_distance = self.max_block_size * 4
    elif shape_type == constants.BOWL:
      min_distance = 0.17
    elif shape_type == constants.PLATE:
      min_distance = 0.2
    else:
      raise ValueError('Attempted to generate invalid shape.')
    return min_distance

  def _getExistingXYPositions(self):
    return [o.getXYPosition() for o in self.objects]

  def _generateShapes(self, shape_type=0, num_shapes=1, scale=None, pos=None, rot=None,
                           min_distance=None, padding=None, random_orientation=False, z_scale=1, model_id=1):
    ''' Generate objects within the workspace.

    Generate a number of objects specified by the shape type within the workspace.

    Args:
      shape_type (int): The shape to generate. Defaults to 0.
      num_shapes (int): The number of objects to generate. Deafults to 1.
      scale (double): The scale for the entire object. Defaults to None.
      pos (list[double]): Generate object at specified position. Defalts to None.
      rot (list[double]): Generate object at specified orientation. Defaults to None,
      min_distance (double): Minimum distance objects must be from other objects. Defaults to None.
      padding (double): Distance from edge of workspace that objects can be generated in. Defaults to None.
      random_orientation (bool): Generates objects with a random orientation if True. Defaults to False.
      z_scale (double): The scale for the objects Z axis. Defaults to 1.
      model_id (int): Used to specify the specific object within a class of object shapes. Defaults to 1.

    Returns:
      list[int] : PyBullet IDs for the generated objects
    '''
    # if padding is not set, use the default padding
    if padding is None:
      padding = self._getDefaultBoarderPadding(shape_type)

    # if self.min_boarder_padding is set, padding can not be smaller than self.min_boarder_padding
    if self.min_boarder_padding is not None:
      padding = max(padding, self.min_boarder_padding)

    # if min_distance is not set, use the default min_distance
    if min_distance is None:
      min_distance = self._getDefaultMinDistance(shape_type)

    # if self.min_object_distance is set, min_distance can not be smaller than self.min_object_distance
    if self.min_object_distance is not None:
      min_distance = max(min_distance, self.min_object_distance)

    shape_handles = list()
    positions = self._getExistingXYPositions()

    if pos is None:
      valid_positions = self._getValidPositions(padding, min_distance, positions, num_shapes)
      if valid_positions is None:
        return None
      for position in valid_positions:
        position.append(self.object_init_z)
    else:
      valid_positions = pos

    if rot is None:
      orientations = [self._getValidOrientation(random_orientation) for _ in range(num_shapes)]
    else:
      orientations = rot


    for position, orientation in zip(valid_positions, orientations):
      if not scale:
        scale = npr.choice(np.arange(self.block_scale_range[0], self.block_scale_range[1]+0.01, 0.02))

      if shape_type == constants.CUBE:
        handle = pb_obj_generation.generateCube(position, orientation, scale)
      elif shape_type == constants.BRICK:
        handle = pb_obj_generation.generateBrick(position, orientation, scale)
      elif shape_type == constants.TRIANGLE:
        handle = pb_obj_generation.generateTriangle(position, orientation, scale)
      elif shape_type == constants.ROOF:
        handle = pb_obj_generation.generateRoof(position, orientation, scale)
      elif shape_type == constants.CYLINDER:
        handle = pb_obj_generation.generateCylinder(position, orientation, scale)
      elif shape_type == constants.RANDOM:
        handle = pb_obj_generation.generateRandomObj(position, orientation, scale, z_scale)
      elif shape_type == constants.CUP:
        handle = pb_obj_generation.generateCup(position, orientation, scale)
      elif shape_type == constants.BOWL:
        handle = pb_obj_generation.generateBowl(position, orientation, scale)
      elif shape_type == constants.PLATE:
        handle = pb_obj_generation.generatePlate(position, orientation, scale, model_id)
      elif shape_type == constants.RANDOM_BLOCK:
        handle = pb_obj_generation.generateRandomBlock(position, orientation, scale)
      elif shape_type == constants.RANDOM_HOUSEHOLD:
        handle = pb_obj_generation.generateRandomHouseHoldObj(position, orientation, scale)
      elif shape_type == constants.SPOON:
        handle = pb_obj_generation.generateSpoon(position, orientation, scale)
      elif shape_type == constants.BOTTLE:
        handle = pb_obj_generation.generateBottle(position, orientation, scale)
      elif shape_type == constants.BOX:
        handle = pb_obj_generation.generateBox(position, orientation, scale)
      elif shape_type == constants.TEST_TUBE:
        handle = pb_obj_generation.generateTestTube(position, orientation, scale, model_id=None)
      elif shape_type == constants.SWAB:
        handle = pb_obj_generation.generateSwab(position, orientation, scale, model_id=None)
      elif shape_type == constants.FLAT_BLOCK:
        handle = pb_obj_generation.generateFlatBlock(position, orientation, scale)
      elif shape_type == constants.RANDOM_HOUSEHOLD200:
        handle = pb_obj_generation.generateRandomHouseHoldObj200(position, orientation, scale, model_id)
      elif shape_type == constants.GRASP_NET_OBJ:
        handle = pb_obj_generation.generateGraspNetObject(position, orientation, scale, model_id)

      else:
        raise NotImplementedError

      if self.physics_mode == 'slow':
        pb.changeDynamics(handle.object_id, -1, linearDamping=0.04, angularDamping=0.04, restitution=0, contactStiffness=3000, contactDamping=100)
      shape_handles.append(handle)
    self.objects.extend(shape_handles)

    for h in shape_handles:
      self.object_types[h] = shape_type

    self.wait(50)
    return shape_handles

  def getObjects(self):
    objs = list()
    for obj in self.objects:
      if self._isObjectHeld(obj):
        continue
      objs.append(obj)
    return objs

  def getObjectPoses(self, objects=None):
    if objects is None: objects = self.objects

    obj_poses = list()
    for obj in objects:
      if self._isObjectHeld(obj):
        continue
      pos, rot = obj.getPose()
      rot = self.convertQuaternionToEuler(rot)

      obj_poses.append(pos + rot)
    return np.array(obj_poses)

  def getObjectPositions(self, omit_hold=True):
    obj_positions = list()
    for obj in self.objects:
      if omit_hold and self._isObjectHeld(obj):
        continue
      obj_positions.append(obj.getPosition())
    return np.array(obj_positions)

  def _getHoldingObj(self):
    return self.robot.holding_obj

  def _getHoldingObjType(self):
    return self.object_types[self._getHoldingObj()] if self.robot.holding_obj else None

  def _isHolding(self):
    return self.robot.holding_obj is not None

  def _getRestPoseMatrix(self):
    T = np.eye(4)
    T[:3, :3] = np.array(pb.getMatrixFromQuaternion(self.rest_pose[1])).reshape((3, 3))
    T[:3, 3] = self.rest_pose[0]
    return T

  def _isObjectHeld(self, obj):
    return self.robot.holding_obj == obj

  def _removeObject(self, obj):
    if obj in self.objects:
      pb.removeBody(obj.object_id)
      # self._moveObjectOutWorkspace(obj)
      self.objects.remove(obj)
      self.robot.openGripper()

  def _moveObjectOutWorkspace(self, obj):
    pos = [-0.50, 0, 0.25]
    obj.resetPose(pos, pb.getQuaternionFromEuler([0., 0., 0.]))

  def _isObjOnTop(self, obj, objects=None):
    if not objects:
      objects = self.objects
    obj_position = obj.getPosition()
    for o in objects:
      if self._isObjectHeld(o) or o is obj:
        continue
      block_position = o.getPosition()
      if np.allclose(block_position[:-1], obj_position[:-1],
                     atol=self.block_original_size * self.block_scale_range[0] * 2 / 3) and \
          block_position[-1] > obj_position[-1]:
        return False
    return True

  def _isObjOnGround(self, obj):
    contact_points = obj.getContactPoints()
    for p in contact_points:
      if p[2] == self.table_id:
        return True
    return False

  def _getNumTopBlock(self, blocks=None):
    if not blocks:
      blocks = self.objects
    cluster_pos = []
    for obj in blocks:
      if self._isObjectHeld(obj):
        continue
      block_position = obj.getPosition()
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

    objects = sorted(objects, key=lambda o: o.getZPosition())
    for i, obj in enumerate(objects):
      if i == 0:
        continue
      #TODO: not 100% sure about this
      if not obj.isTouching(objects[i-1]) or obj.isTouching(PybulletObject(-1, self.table_id)):
        return False
      # the xy positions of the blocks much be close. Otherwise the adjacent blocks will be considered as a stack
      if not np.allclose(np.array(obj.getXYPosition()), np.array(objects[i-1].getXYPosition()), atol=self.max_block_size/2):
        return False
    return True

  def _checkPerfectGrasp(self, x, y, z, rot, objects):
    '''
    check if the grasp at (x, y, z, rot) is a perfect grasp (gripper's rotation aligned with the object's rotation)
    this method only checks rz, and only works for basic block stacking tasks
    :param x: x of the grasp pose
    :param y: y of the grasp pose
    :param z: z of the grasp pose
    :param rot: rotation of the grasp pose
    :param objects: candidate objects to grasp
    :return: True if the relative rotation between the gripper and the target object is smaller than pi/12
    '''
    end_pos = np.array([x, y, z])
    sorted_obj = sorted(objects, key=lambda o: np.linalg.norm(end_pos - o.getPosition()))
    obj_pos, obj_rot = sorted_obj[0].getPose()
    obj_type = self.object_types[sorted_obj[0]]
    # For cylinders, the gripper is always aligned
    if obj_type is constants.CYLINDER:
      return True
    obj_rot = pb.getEulerFromQuaternion(obj_rot)
    # The relative angle between the gripper and the target object in [0, pi] range
    angle = np.pi - np.abs(np.abs(rot[2] - obj_rot[2]) - np.pi)
    # Cubes are symmetric for pi/2 rotation
    if obj_type is constants.CUBE:
      while angle > np.pi / 2:
        angle -= np.pi / 2
      angle = min(angle, np.pi / 2 - angle)
    # Bricks, roofs, and triangles are symmetric for pi rotation
    elif obj_type in [constants.BRICK, constants.ROOF, constants.TRIANGLE]:
      angle = min(angle, np.pi - angle)
    return angle < np.pi / 12

  def _checkPerfectPlace(self, x, y, z, rot, objects):
    '''
    check if the place at (x, y, z, rot) is a perfect place (gripper's rotation aligned with the object's rotation)
    this method only checks rz, and only works for basic block stacking tasks
    :param x: x of the place pose
    :param y: y of the place pose
    :param z: z of the place pose
    :param rot: rotation of the place pose
    :param objects: candidate objects to place on
    :return: True if the relative rotation between the gripper and the target object is smaller than pi/12
    '''
    end_pos = np.array([x, y, z])
    sorted_obj = sorted(objects, key=lambda o: np.linalg.norm(end_pos - o.getPosition()))
    obj_pos, obj_rot = sorted_obj[0].getPose()
    holding_obj_type = self._getHoldingObjType()
    obj_rot = pb.getEulerFromQuaternion(obj_rot)
    angle = np.pi - np.abs(np.abs(rot[2] - obj_rot[2]) - np.pi)
    if angle > np.pi/2:
      angle -= np.pi/2
    angle = min(angle, np.pi / 2 - angle)
    return angle < np.pi / 12

  def _checkObjUpright(self, obj, threshold=np.pi/9):
    triangle_rot = obj.getRotation()
    triangle_rot = pb.getEulerFromQuaternion(triangle_rot)
    return abs(triangle_rot[0]) < threshold and abs(triangle_rot[1]) < threshold

  def _checkOnTop(self, bottom_obj, top_obj):
    bottom_position = bottom_obj.getPosition()
    top_position = top_obj.getPosition()
    if top_position[-1] - bottom_position[-1] < 0.5 * self.block_scale_range[0] * self.block_original_size:
      return False
    return top_obj.isTouching(bottom_obj)

  def _getDistance(self, obj1, obj2):
    position1 = obj1.getPosition()
    position2 = obj2.getPosition()
    return np.linalg.norm(np.array(position1) - np.array(position2))

  def _checkAdjacent(self, obj_1, obj_2):
    return np.allclose(obj_1.getZPosition(), obj_2.getZPosition(), atol=0.01) and \
           self._getDistance(obj_1, obj_2) < 2.2 * self.max_block_size

  def _checkInBetween(self, obj0, obj1, obj2, threshold=None):
    if not threshold:
      threshold = self.max_block_size
    position0 = obj0.getXYPosition()
    position1 = obj1.getXYPosition()
    position2 = obj2.getXYPosition()
    middle_point = np.mean((np.array(position1), np.array(position2)), axis=0)
    dist = np.linalg.norm(middle_point - position0)
    return dist < threshold

  def _checkOriSimilar(self, objects, threshold=np.pi/7):
    oris = list(map(lambda o: pb.getEulerFromQuaternion(o.getRotation())[2], objects))
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
      if self.object_types[self.robot.holding_obj] is constants.BRICK:
        extend = int(2*self.max_block_size/self.heightmap_resolution)
      elif self.object_types[self.robot.holding_obj] is constants.ROOF:
        extend = int(1.5*self.max_block_size/self.heightmap_resolution)
      else:
        extend = int(0.5*self.max_block_size/self.heightmap_resolution)
    else:
      extend = int(0.5*self.max_block_size/self.heightmap_resolution)
    local_region = self.heightmap[int(max(x_pixel - extend, 0)):int(min(x_pixel + extend, self.heightmap_size)), \
                                  int(max(y_pixel - extend, 0)):int(min(y_pixel + extend, self.heightmap_size))]
    try:
      safe_z_pos = np.max(local_region) + self.workspace[2][0]
    except ValueError:
      safe_z_pos = self.workspace[2][0]
    if motion_primative == constants.PICK_PRIMATIVE:
      safe_z_pos -= self.pick_offset
      safe_z_pos = max(safe_z_pos, 0.01)
    else:
      safe_z_pos += self.place_offset
    return safe_z_pos

  def convertQuaternionToEuler(self, rot):
    rot = list(pb.getEulerFromQuaternion(rot))

    # TODO: This normalization should be improved
    # while rot[2] < 0:
    #   rot[2] += np.pi
    # while rot[2] > np.pi:
    #   rot[2] -= np.pi

    return rot

  def normalizeRot(self, rx, ry, rz):
    while rz < 0:
      rz += 2*np.pi
    while rz > 2*np.pi:
      rz -= 2*np.pi

    if self.half_rotation:
      if rz > np.pi:
        rz -= np.pi
        rx = -rx
        ry = -ry
      if type(self.robot) is Kuka:
        rz -= np.pi
        rx = -rx
        ry = -ry
    return rx, ry, rz

  def _decodeAction(self, action):
    """
    decode input action base on self.action_sequence
    Args:
      action: action tensor

    Returns: motion_primative, x, y, z, rot

    """
    primative_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a), ['p', 'x', 'y', 'z', 'r'])
    motion_primative = action[primative_idx] if primative_idx != -1 else 0
    x = action[x_idx]
    y = action[y_idx]
    z = action[z_idx] if z_idx != -1 else self._getPrimativeHeight(motion_primative, x, y)
    rz, ry, rx = 0, 0, 0
    if self.action_sequence.count('r') <= 1:
      rz = action[rot_idx] if rot_idx != -1 else 0
      ry = 0
      rx = 0
    elif self.action_sequence.count('r') == 2:
      rz = action[rot_idx]
      ry = 0
      rx = action[rot_idx+1]
    elif self.action_sequence.count('r') == 3:
      rz = action[rot_idx]
      ry = action[rot_idx + 1]
      rx = action[rot_idx + 2]

    rot = (rx, ry, rz)

    return motion_primative, x, y, z, rot

  def _encodeAction(self, primitive, x, y, z, r):
    if hasattr(r, '__len__'):
      assert len(r) in [1, 2, 3]
      if len(r) == 1:
        rz = r[0]
        ry = 0
        rx = 0
      elif len(r) == 2:
        rz = r[0]
        ry = 0
        rx = r[1]
      else:
        rz = r[0]
        ry = r[1]
        rx = r[2]
    else:
      rz = r
      ry = 0
      rx = 0
    while rz > np.pi * 2:
      rz -= np.pi * 2
    while rz < 0:
      rz += np.pi * 2

    primitive_idx, x_idx, y_idx, z_idx, rot_idx = map(lambda a: self.action_sequence.find(a),
                                                      ['p', 'x', 'y', 'z', 'r'])
    action = np.zeros(len(self.action_sequence), dtype=float)
    if primitive_idx != -1:
      action[primitive_idx] = primitive
    if x_idx != -1:
      action[x_idx] = x
    if y_idx != -1:
      action[y_idx] = y
    if z_idx != -1:
      action[z_idx] = z
    if rot_idx != -1:
      if self.action_sequence.count('r') == 1:
        action[rot_idx] = rz
      elif self.action_sequence.count('r') == 2:
        action[rot_idx] = rz
        action[rot_idx+1] = rx
      elif self.action_sequence.count('r') == 3:
        action[rot_idx] = rz
        action[rot_idx+1] = ry
        action[rot_idx+2] = rx

    return action

  def _checkTermination(self):
    '''
    Sub-envs should override this to set their own termination conditions
    Returns: False
    '''
    return False

  def _getShapeName(self, shape_type):
    ''' Get the shape name from the type (int) '''
    if shape_type == self.CUBE: return 'cube'
    elif shape_type == self.SPHERE: return 'sphere'
    elif shape_type == self.CYLINDER: return 'cylinder'
    elif shape_type == self.CONE: return 'cone'
    elif shape_type == self.BRICK: return 'brick'
    else: return 'unknown'

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

    return round(x_pixel), round(y_pixel)

  def _getPosFromPixels(self, x_pixel, y_pixel):
    x = x_pixel * self.heightmap_resolution + self.workspace[0][0]
    y = y_pixel * self.heightmap_resolution + self.workspace[1][0]
    return x, y

  def _isObjectOnCandidatePose(self, obj):
    '''
    Checks if the object has drifted off the candidate positions.
    Args:
      - obs: A simulated object
    Returns: True if object is close to a candidate position, False otherwise
    '''
    pos = obj.getPosition()
    return np.abs(self.pos_candidate[0] - pos[0]).min() > 0.02 or \
           np.abs(self.pos_candidate[1] - pos[1]).min() > 0.02

  def _isObjectWithinWorkspace(self, obj):
    '''
    Checks if the object is entirely within the workspace.
    Args:
      - obs: A simulated object
    Returns: True if bounding box of object is within workspace, False otherwise
    '''
    xyz_min, xyz_max = obj.getBoundingBox()
    # TODO: Need to always use max z value as min z value is just under zero
    p1 = [xyz_min[0], xyz_min[1], xyz_max[2]]
    p2 = [xyz_min[0], xyz_max[1], xyz_max[2]]
    p3 = [xyz_max[0], xyz_min[1], xyz_max[2]]
    p4 = xyz_max

    return all(map(lambda x: self._isPointInWorkspace(x), [p1, p2, p3, p4]))

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

  def getInHandImage(self, heightmap, x, y, z, rot, next_heightmap):
    (rx, ry, rz) = rot
    # Pad heightmaps for grasps near the edges of the workspace
    heightmap = np.pad(heightmap, int(self.in_hand_size / 2), 'constant', constant_values=0.0)
    next_heightmap = np.pad(next_heightmap, int(self.in_hand_size / 2), 'constant', constant_values=0.0)

    x, y = self._getPixelsFromPos(x, y)
    x = x + int(self.in_hand_size / 2)
    y = y + int(self.in_hand_size / 2)

    # Get the corners of the crop
    x_min = int(x - self.in_hand_size / 2)
    x_max = int(x + self.in_hand_size / 2)
    y_min = int(y - self.in_hand_size / 2)
    y_max = int(y + self.in_hand_size / 2)

    # Crop both heightmaps
    crop = heightmap[x_min:x_max, y_min:y_max]
    if self.in_hand_mode.find('sub') > -1:
      next_crop = next_heightmap[x_min:x_max, y_min:y_max]

      # Adjust the in-hand image to remove background objects
      next_max = np.max(next_crop)
      crop[crop >= next_max] -= next_max

    if self.in_hand_mode.find('proj') > -1:
      return self.getInHandOccupancyGridProj(crop, z, rot)
    else:
      # end_effector rotate counter clockwise along z, so in hand img rotate clockwise
      crop = sk_transform.rotate(crop, np.rad2deg(-rz))
      return crop.reshape((1, self.in_hand_size, self.in_hand_size))

  def getInHandOccupancyGridProj(self, crop, z, rot):
    rx, ry, rz = rot
    # crop = zoom(crop, 2)
    crop = np.round(crop, 5)
    size = self.in_hand_size

    zs = np.array([z+(-size/2+i)*(self.heightmap_resolution) for i in range(size)])
    zs = zs.reshape((1, 1, -1))
    zs = zs.repeat(size, 0).repeat(size, 1)
    c = crop.reshape(size, size, 1).repeat(size, 2)
    ori_occupancy = c > zs

    # transform into points
    point = np.argwhere(ori_occupancy)
    if point.shape[0] == 0:
      return np.zeros((3, size, size))

    # center
    ori_point = point - size/2
    R = transformations.euler_matrix(rx, ry, rz)[:3, :3].T
    points = []
    batch_size = 4096
    for i in range(int(np.ceil(ori_point.shape[0]/batch_size))):
      points.append(R.dot(ori_point[batch_size*i:batch_size*(i+1)].T))
    point = np.concatenate(points, 1)
    # point = R.dot(ori_point.T)
    point = point + size/2
    point = np.round(point).astype(int)
    point = point.T[(np.logical_and(0 < point.T, point.T < size)).all(1)].T

    occupancy = np.zeros((size, size, size))
    occupancy[point[0], point[1], point[2]] = 1
    occupancy = median_filter(occupancy, size=2)
    occupancy = np.ceil(occupancy)

    projection = np.stack((occupancy.sum(0), occupancy.sum(1), occupancy.sum(2)))
    return projection

  def getEmptyInHand(self):
    if self.in_hand_mode.find('proj') > -1:
      return np.zeros((3, self.in_hand_size, self.in_hand_size))
    else:
      return np.zeros((1, self.in_hand_size, self.in_hand_size))

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

  def getStateDict(self):
    self.robot.saveState()
    state = {'current_episode_steps': copy.deepcopy(self.current_episode_steps),
             'objects': copy.deepcopy(self.objects),
             'object_types': copy.deepcopy(self.object_types),
             'heightmap': copy.deepcopy(self.heightmap),
             'robot_state': copy.deepcopy(self.robot.state),
             'random_state': np.random.get_state(),
             'last_action': self.last_action,
             'last_obj': self.last_obj
             }
    return state

  def restoreStateDict(self, state):
    self.current_episode_steps = state['current_episode_steps']
    self.objects = state['objects']
    self.object_types = state['object_types']
    self.heightmap = state['heightmap']
    self.last_action = state['last_action']
    self.last_obj = state['last_obj']
    self.robot.state = state['robot_state']
    self.robot.restoreState()
    np.random.set_state(state['random_state'])

  def saveState(self):
    self.pb_state = pb.saveState()
    self.state = self.getStateDict()

  def restoreState(self):
    pb.restoreState(self.pb_state)
    self.restoreStateDict(self.state)

  def saveEnvToFile(self, path):
    bullet_file = os.path.join(path, 'env.bullet')
    pickle_file = os.path.join(path, 'env.pickle')
    pb.saveBullet(bullet_file)
    state = self.getStateDict()
    with open(pickle_file, 'wb') as f:
      pickle.dump(state, f)

  def loadEnvFromFile(self, path):
    bullet_file = os.path.join(path, 'env.bullet')
    pickle_file = os.path.join(path, 'env.pickle')
    pb.restoreState(fileName=bullet_file)
    with open(pickle_file, 'rb') as f:
      state = pickle.load(f)
    self.restoreStateDict(state)

