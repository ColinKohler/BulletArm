import pybullet as pb
import numpy as np
import scipy
import numpy.random as npr
from itertools import combinations

from bulletarm.envs.base_env import BaseEnv, NoValidPositionException
import bulletarm.pybullet.utils.object_generation as pb_obj_generation
from bulletarm.pybullet.utils import constants

class DeconstructEnv(BaseEnv):
  '''

  '''
  def __init__(self, config):
    super(DeconstructEnv, self).__init__(config)
    self.pick_offset = 0.01
    self.terminate_min_dist = 2.4*self.min_block_size
    self.structure_objs = list()
    self.prev_obj_pos = None

  def takeAction(self, action):
    # keep track of the current positions of all objects
    self.prev_obj_pos = self.getObjectPositions(omit_hold=False)
    BaseEnv.takeAction(self, action)

  def isSimValid(self):
    if self.prev_obj_pos is None:
      return True
    else:
      # Compare the object positions with the previous step. Only allow one object to move at each action step
      curr_obj_pos = self.getObjectPositions(omit_hold=False)
      dist = np.linalg.norm(curr_obj_pos - self.prev_obj_pos, axis=1)
      return (dist > 0.005).sum() == 1 and BaseEnv.isSimValid(self)

  def _getObservation(self, action=None):
    '''
    In deconstruct, get the in-hand image after placing an object since deconstruction is the reverse of construction
    '''
    old_heightmap = self.heightmap
    self.heightmap = self._getHeightmap()

    if action is None or self._isHolding() == True:
      in_hand_img = self.getEmptyInHand()
    else:
      motion_primative, x, y, z, rot = self._decodeAction(action)
      in_hand_img = self.getInHandImage(self.heightmap, x, y, z, rot, old_heightmap)

    return self._isHolding(), in_hand_img, self.heightmap.reshape([1, self.heightmap_size, self.heightmap_size])

  def resetDeconstructEnv(self):
    self.resetPybulletWorkspace()
    self.structure_objs = list()
    self.generateStructure()
    while not self.checkStructure():
      self.resetPybulletWorkspace()
      self.structure_objs = list()
      self.generateStructure()

  def reset(self):
    ''''''
    self.resetDeconstructEnv()
    return self._getObservation()

  def _checkTermination(self):
    # To deconstruct a block structure with n blocks, n-1 pick and place action pairs must be executed
    if self.current_episode_steps < (self.num_obj-1)*2:
      return False
    obj_combs = combinations(self.objects, 2)
    for (obj1, obj2) in obj_combs:
      dist = np.linalg.norm(np.array(obj1.getXYPosition()) - np.array(obj2.getXYPosition()))
      if dist < self.terminate_min_dist:
        return False
    return True

  def checkStructure(self):
    raise NotImplemented('Deconstruct env must implement this function')

  def generateStructure(self):
    raise NotImplemented('Deconstruct env must implement this function')

  def step(self, action):
    reward = 1.0 if self.checkStructure() else 0.0
    self.takeAction(action)
    self.wait(100)
    obs = self._getObservation(action)
    motion_primative, x, y, z, rot = self._decodeAction(action)
    done = motion_primative and self._checkTermination()

    if not done:
      done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
    self.current_episode_steps += 1

    return obs, reward, done

  def generateStructureShape(self, pos, rot, obj_type, scale=None):
    '''
    '''
    # add the random initialization offset
    x_offset = (np.random.random()-1) * self.deconstruct_init_offset
    y_offset = (np.random.random()-1) * self.deconstruct_init_offset
    pos = list(pos)
    pos[0] += x_offset
    pos[1] += y_offset

    if scale is None:
      scale = npr.choice(np.arange(self.block_scale_range[0], self.block_scale_range[1] + 0.01, 0.02))

    if obj_type == constants.CUBE:
      handle = pb_obj_generation.generateCube(pos, rot, scale)
    elif obj_type == constants.BRICK:
      handle = pb_obj_generation.generateBrick(pos, rot, scale)
    elif obj_type == constants.ROOF:
      handle = pb_obj_generation.generateRoof(pos, rot, scale)
    elif obj_type == constants.TRIANGLE:
      handle = pb_obj_generation.generateTriangle(pos, rot, scale)
    elif obj_type == constants.CYLINDER:
      handle = pb_obj_generation.generateCylinder(pos, rot, scale)
    elif obj_type == constants.RANDOM:
      handle = pb_obj_generation.generateRandomObj(pos, rot, scale)
    else:
      raise NotImplementedError
    self.objects.append(handle)
    self.object_types[handle] = obj_type
    self.structure_objs.append(handle)
    return handle

  def generateStructureRandomShapeWithZScale(self, pos, rot, zscale=1):
    handle = pb_obj_generation.generateRandomObj(pos, rot,
                                                 npr.uniform(self.block_scale_range[0], self.block_scale_range[1]),
                                                 zscale)
    self.objects.append(handle)
    self.object_types[handle] = constants.RANDOM
    self.structure_objs.append(handle)

  def generateStructureRandomShapeWithScaleAndZScale(self, pos, rot, scale, zscale):
    handle = pb_obj_generation.generateRandomObj(pos, rot, scale, zscale)
    self.objects.append(handle)
    self.object_types[handle] = constants.RANDOM
    self.structure_objs.append(handle)

  def generateStructureRandomBrickShape(self, pos, rot, x_scale=0.6, y_scale=0.6, z_scale=0.6):
    handle = pb_obj_generation.generateRandomBrick(pos, rot, x_scale, y_scale, z_scale)
    self.objects.append(handle)
    self.object_types[handle] = constants.BRICK
    self.structure_objs.append(handle)

  def get1BaseXY(self, padding):
    return self._getValidPositions(padding, 0, [], 1)[0]

  def get2BaseXY(self, padding, min_dist, max_dist):
    pos1 = self._getValidPositions(padding, 0, [], 1)[0]
    if self.random_orientation:
      sample_range = [[pos1[0] - max_dist, pos1[0] + max_dist],
                      [pos1[1] - max_dist, pos1[1] + max_dist]]
    else:
      sample_range = [[pos1[0] - 0.005, pos1[0] + 0.005],
                      [pos1[1] - max_dist, pos1[1] + max_dist]]
    for i in range(100):
      try:
        pos2 = self._getValidPositions(padding, min_dist, [pos1], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
      if min_dist < dist < max_dist:
        break
    return pos1, pos2

  def getXYRFrom2BasePos(self, pos1, pos2):
    obj_positions = np.array([pos1, pos2])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y = obj_positions.mean(0)
    r = np.arctan(slope) + np.pi / 2 if self.random_orientation else 0
    while r > np.pi:
      r -= np.pi
    return x, y, r
