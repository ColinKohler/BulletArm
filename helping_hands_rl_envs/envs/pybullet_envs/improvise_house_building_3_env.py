from copy import deepcopy
import numpy as np
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import object_generation
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util
from helping_hands_rl_envs.simulators.constants import NoValidPositionException

class ImproviseHouseBuilding3Env(PyBulletEnv):
  ''''''
  def __init__(self, config):
    config['check_random_obj_valid'] = True
    super(ImproviseHouseBuilding3Env, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletEnv()
      try:
        padding = pybullet_util.getPadding(constants.BRICK, self.max_block_size)
        min_distance = pybullet_util.getMinDistance(constants.BRICK, self.max_block_size)
        pos = self._getValidPositions(padding, min_distance, [], 1)[0]
        rot = self._getValidOrientation(self.random_orientation)
        brick_xscale = np.random.uniform(0.5, 0.7)
        brick_yscale = np.random.uniform(0.5, 0.7)
        brick_zscale = np.random.uniform(0.4, 0.7)
        handle = object_generation.generateRandomBrick(pos, rot, brick_xscale, brick_yscale, brick_zscale)
        self.objects.append(handle)
        self.object_types[handle] = constants.BRICK

        self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)

        for i in range(2):
          zscale = np.random.uniform(2, 2.2)
          scale = np.random.uniform(0.6, 0.9)
          zscale = 0.6 * zscale / scale
          self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation, scale=scale, z_scale=zscale)
      except NoValidPositionException:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
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

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(ImproviseHouseBuilding3Env, self).isSimValid()

def createImproviseHouseBuilding3Env(config):
  return ImproviseHouseBuilding3Env(config)
