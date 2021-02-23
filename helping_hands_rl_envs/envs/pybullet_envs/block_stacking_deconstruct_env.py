import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_envs.deconstruct_env import DeconstructEnv
from helping_hands_rl_envs.simulators import constants

class BlockStackingDeconstructEnv(DeconstructEnv):
  ''''''
  def __init__(self, config):
    super(BlockStackingDeconstructEnv, self).__init__(config)

  def checkStructure(self):
    ''''''
    return self._checkStack()

  def generateStructure(self):
    padding = self.max_block_size * 1.5
    pos = self.get1BaseXY(padding)
    rot = self._getValidOrientation(self.random_orientation)
    for i in range(self.num_obj):
      self.generateStructureShape((pos[0], pos[1], i * self.max_block_size + self.max_block_size / 2), rot,
                                  constants.CUBE)
    self.wait(50)

def createHouseBuilding1DeconstructEnv(config):
  return BlockStackingDeconstructEnv(config)
