import numpy as np
from helping_hands_rl_envs.envs.pybullet_envs.ramp_envs.ramp_base_env import RampBaseEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import object_generation


class RampImproviseHouseBuilding3Env(RampBaseEnv):
  def __init__(self, config):
    super(RampImproviseHouseBuilding3Env, self).__init__(config)

  def generateRandomShape(self, n, poss, rots):
    for i in range(n):
      zscale = np.random.uniform(2, 2.2)
      scale = np.random.uniform(0.6, 0.9)
      zscale = 0.6 * zscale / scale
      self._generateShapes(constants.RANDOM, 1, random_orientation=False, pos=poss[i:i + 1], rot=rots[i:i + 1],
                           scale=scale, z_scale=zscale)

  def generateBrickShape(self, n, poss, rots):
    for i in range(n):
      brick_xscale = np.random.uniform(0.5, 0.7)
      brick_yscale = np.random.uniform(0.5, 0.7)
      brick_zscale = np.random.uniform(0.4, 0.7)
      handle = object_generation.generateRandomBrick(poss[i], rots[i], brick_xscale, brick_yscale, brick_zscale)
      self.objects.append(handle)
      self.object_types[handle] = constants.BRICK

  def reset(self):
    obj_dict = {
      constants.ROOF: 1,
      constants.BRICK: 1,
      constants.RANDOM: 2
    }
    self.resetWithRampAndObj(obj_dict)
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

def createRampImproviseHouseBuilding3Env(config):
  return RampImproviseHouseBuilding3Env(config)