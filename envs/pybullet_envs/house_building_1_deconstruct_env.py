import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_envs.deconstruct_env import DeconstructEnv
from helping_hands_rl_envs.simulators import constants

class HouseBuilding1DeconstructEnv(DeconstructEnv):
  ''''''
  def __init__(self, config):
    super(HouseBuilding1DeconstructEnv, self).__init__(config)

  def reset(self):
    ''''''
    super(HouseBuilding1DeconstructEnv, self).reset()
    self.generateH1()

    while not self.checkStructure():
      super(HouseBuilding1DeconstructEnv, self).reset()
      self.generateH1()

    return self._getObservation()

  def _checkTermination(self):
    obj_combs = combinations(self.objects, 2)
    for (obj1, obj2) in obj_combs:
      dist = np.linalg.norm(np.array(obj1.getXYPosition()) - np.array(obj2.getXYPosition()))
      if dist < 2.4*self.min_block_size:
        return False
    return True

  def checkStructure(self):
    ''''''
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return self._checkStack(blocks+triangles) and self._checkObjUpright(triangles[0])

  def isSimValid(self):
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return self._checkObjUpright(triangles[0]) and super(HouseBuilding1DeconstructEnv, self).isSimValid()


def createHouseBuilding1DeconstructEnv(config):
  def thunk():
    return HouseBuilding1DeconstructEnv(config)
  return thunk