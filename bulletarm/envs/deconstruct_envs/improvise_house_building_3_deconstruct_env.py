import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
import pybullet as pb
from itertools import combinations
from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.pybullet.utils import constants

class ImproviseHouseBuilding3DeconstructEnv(DeconstructEnv):
  ''''''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 4
    if 'max_steps' not in config:
      config['max_steps'] = 10
    config['check_random_obj_valid'] = True
    super(ImproviseHouseBuilding3DeconstructEnv, self).__init__(config)

  def checkStructure(self):
    random_blocks = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    top_blocks = []
    for block in random_blocks:
      if self._isObjOnTop(block, random_blocks):
        top_blocks.append(block)
    if len(top_blocks) != 2:
      return False
    if self._checkOnTop(top_blocks[0], bricks[0]) and \
        self._checkOnTop(top_blocks[1], bricks[0]) and \
        self._checkOnTop(bricks[0], roofs[0]) and \
        self._checkOriSimilar([bricks[0], roofs[0]]) and \
        self._checkInBetween(bricks[0], random_blocks[0], random_blocks[1]) and \
        self._checkInBetween(roofs[0], random_blocks[0], random_blocks[1]):
      return True
    return False

  def generateStructure(self):
    padding = self.max_block_size * 1.5
    min_dist = 2.7 * self.max_block_size
    max_dist = 2.8 * self.max_block_size
    pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)

    zscale1 = np.random.uniform(2, 2.2)
    scale1 = np.random.uniform(0.6, 0.9)
    zscale1 = 0.6 * zscale1 / scale1

    zscale2 = np.random.uniform(2, 2.2)
    scale2 = np.random.uniform(0.6, 0.9)
    zscale2 = 0.6 * zscale2 / scale2

    brick_xscale = np.random.uniform(0.5, 0.7)
    brick_yscale = np.random.uniform(0.5, 0.7)
    brick_zscale = np.random.uniform(0.4, 0.7)

    self.generateStructureRandomShapeWithScaleAndZScale([pos1[0], pos1[1], zscale1 * 0.007], self._getValidOrientation(self.random_orientation), scale1, zscale1)
    self.generateStructureRandomShapeWithScaleAndZScale([pos2[0], pos2[1], zscale2 * 0.007], self._getValidOrientation(self.random_orientation), scale2, zscale2)

    x, y, r = self.getXYRFrom2BasePos(pos1, pos2)

    self.generateStructureRandomBrickShape((x, y, self.max_block_size * 1.5), pb.getQuaternionFromEuler([0., 0., r]),
                                           brick_xscale, brick_yscale, brick_zscale)

    self.generateStructureShape((x, y, self.max_block_size * 2.5), pb.getQuaternionFromEuler([0., 0., r]), constants.ROOF)
    self.wait(50)

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and DeconstructEnv.isSimValid(self)

def createImproviseHouseBuilding3DeconstructEnv(config):
  return ImproviseHouseBuilding3DeconstructEnv(config)
