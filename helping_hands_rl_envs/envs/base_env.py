import numpy as np
import numpy.random  as npr
from scipy.ndimage import median_filter
import skimage.transform as sk_transform

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

import matplotlib.pyplot as plt

class BaseEnv(object):
  '''
  Base Arm RL environment.
  '''
  def __init__(self, seed, workspace, max_steps, heightmap_size, action_sequence='pxyr', pos_candidate=None,
               in_hand_size=24, in_hand_mode='sub'):
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
    self.seed = seed
    npr.seed(self.seed)

    self.active_env_id = 0

    # Setup environment
    self.workspace = workspace
    self.workspace_size = np.linalg.norm(self.workspace[0,1] - self.workspace[0,0])
    self.pos_candidate = pos_candidate.astype(np.int) if pos_candidate else pos_candidate
    self.max_steps = max_steps

    # Setup heightmap
    self.heightmap_size = heightmap_size
    self.in_hand_size = in_hand_size
    self.in_hand_mode = in_hand_mode
    self.heightmap_shape = (self.heightmap_size, self.heightmap_size, 1)
    self.heightmap_resolution = self.workspace_size / self.heightmap_size

    assert action_sequence.find('x') != -1
    assert action_sequence.find('y') != -1
    self.action_sequence = action_sequence

    # Setup observation and action spaces
    self.obs_shape = self.heightmap_shape
    self.num_primatives = constants.NUM_PRIMATIVES

    # TODO: This is a bit of a hacky way to set this but I'm not sure if there is a better way due to the
    #       action seqeuence stuff
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

    self.offset = 0.01

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
    action = np.zeros(len(self.action_sequence), dtype=np.float)
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
    safe_z_pos = np.max(local_region) + self.workspace[2][0]
    safe_z_pos = safe_z_pos - self.offset if motion_primative == constants.PICK_PRIMATIVE else safe_z_pos + self.offset

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
    x = np.clip(x, self.in_hand_size / 2, self.heightmap_size-1-self.in_hand_size/2)
    y = np.clip(y, self.in_hand_size / 2, self.heightmap_size-1-self.in_hand_size/2)
    x = round(x + self.in_hand_size / 2)
    y = round(y + self.in_hand_size / 2)

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
    # zs[zs<-(self.heightmap_resolution)] = 100
    c = crop.reshape(size, size, 1).repeat(size, 2)
    ori_occupancy = c > zs

    # transform into points
    point = np.argwhere(ori_occupancy)
    # center
    ori_point = point - size/2
    R = transformations.euler_matrix(rx, ry, rz)[:3, :3].T
    point = R.dot(ori_point.T)
    point = point + size/2
    point = np.round(point).astype(int)
    point = point.T[(np.logical_and(0 < point.T, point.T < size)).all(1)].T

    occupancy = np.zeros((size, size, size))
    occupancy[point[0], point[1], point[2]] = 1
    occupancy = median_filter(occupancy, size=2)
    occupancy = np.ceil(occupancy)

    projection = np.stack((occupancy.sum(0), occupancy.sum(1), occupancy.sum(2)))
    projection = np.rollaxis(projection, 0, 3)
    # fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    # axs[0].imshow(projection[:, :, 0])
    # axs[1].imshow(projection[:, :, 1])
    # axs[2].imshow(projection[:, :, 2])
    # fig.show()
    return projection

  def getEmptyInHand(self):
    if self.in_hand_mode.find('proj') > -1:
      return np.zeros((3, self.in_hand_size, self.in_hand_size))
    else:
      return np.zeros((1, self.in_hand_size, self.in_hand_size))
