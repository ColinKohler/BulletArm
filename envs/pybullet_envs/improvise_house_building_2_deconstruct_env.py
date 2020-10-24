import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
import pybullet as pb
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_envs.deconstruct_env import DeconstructEnv
from helping_hands_rl_envs.simulators import constants

class ImproviseHouseBuilding2Env(DeconstructEnv):
  ''''''
  def __init__(self, config):
    config['check_random_obj_valid'] = True
    super(ImproviseHouseBuilding2Env, self).__init__(config)
    self.terminate_min_dist = 2.7*self.min_block_size

  def checkStructure(self):
    rand_objs = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    rand_obj_combs = combinations(rand_objs, 2)
    for (obj1, obj2) in rand_obj_combs:
      if self._checkOnTop(obj1, roofs[0]) and self._checkOnTop(obj2, roofs[0]):
        return True
    return False

  def generateStructure(self):
    lower_z1 = 0.01
    roof_z = 0.025

    padding = self.max_block_size * 1.5
    min_dist = 1.7 * self.max_block_size
    max_dist = 2.4 * self.max_block_size
    pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)

    self.generateStructureShape([pos1[0], pos1[1], lower_z1], self._getValidOrientation(self.random_orientation), constants.RANDOM)
    self.generateStructureShape([pos2[0], pos2[1], lower_z1], self._getValidOrientation(self.random_orientation), constants.RANDOM)

    x, y, r = self.getXYRFrom2BasePos(pos1, pos2)

    self.generateStructureShape((x, y, roof_z), pb.getQuaternionFromEuler([0., 0., r]), constants.ROOF)
    self.wait(50)

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(ImproviseHouseBuilding2Env, self).isSimValid()

def createImproviseHouseBuilding2DeconstructEnv(config):
  return ImproviseHouseBuilding2Env(config)
