import numpy as np
import numpy.random  as npr

class BaseEnv(object):
  '''
  Base Arm RL environment.
  '''
  def __init__(self, seed, workspace, max_steps, heightmap_size, action_sequence='pxyr'):
    """
    constructor of BaseEnv
    Args:
      seed: random seed used in this environment
      workspace: workspace of the arm
      max_steps: maximum number of steps in an episode
      heightmap_size: size of observation heightmap
      action_sequence: the format of input action in each step. It is a sequence of 'pxyzr'. Must include 'x' and 'y'
                       eg: action_sequence = 'pxyr' => motion_primative, x, y, rot = action

    """
    # Set random numpy seed
    npr.seed(seed)

    # Setup environment
    self.workspace = workspace
    self.workspace_size = np.linalg.norm(self.workspace[0,1] - self.workspace[0,0])
    self.max_steps = max_steps

    # Setup heightmap
    self.heightmap_size = heightmap_size
    self.heightmap_shape = (self.heightmap_size, self.heightmap_size, 1)
    self.heightmap_resolution = self.workspace_size / self.heightmap_size

    # Setup observation and action spaces
    self.obs_shape = self.heightmap_shape
    self.action_space = np.concatenate((self.workspace[:2,:], np.array([[0.0], [2*np.pi]])), axis=1)
    self.action_shape = 3

    # Motion primatives
    self.PICK_PRIMATIVE = 0
    self.PLACE_PRIMATIVE = 1
    self.PUSH_PRIMATIVE = 2

    # Shape types
    self.CUBE = 0
    self.SPHERE = 1
    self.CYLINDER = 2
    self.CONE = 3

    assert action_sequence.find('x') != -1
    assert action_sequence.find('y') != -1
    self.action_sequence = action_sequence

  def _getSpecificAction(self, action):
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
    rot = action[rot_idx] if rot_idx != -1 else 0
    return motion_primative, x, y, z, rot

  def _getShapeName(self, shape_type):
    ''' Get the shape name from the type (int) '''
    if shape_type == self.CUBE: return 'cube'
    elif shape_type == self.SPHERE: return 'sphere'
    elif shape_type == self.CYLINER: return 'cylinder'
    elif shape_type == self.CONE: return 'cone'
    else: return 'unknown'

  def _getPrimativeHeight(self, motion_primative, x, y, offset=0.01):
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
    local_region = self.heightmap[max(y_pixel - 30, 0):min(y_pixel + 30, self.heightmap_size), \
                                  max(x_pixel - 30, 0):min(x_pixel + 30, self.heightmap_size)]
    safe_z_pos = np.max(local_region) + self.workspace[2][0]
    safe_z_pos = safe_z_pos - offset if motion_primative == self.PICK_PRIMATIVE else safe_z_pos + offset

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

  def _checkTermination(self):
    '''
    Sub-envs should override this to set their own termination conditions
    Returns: False
    '''
    return False

