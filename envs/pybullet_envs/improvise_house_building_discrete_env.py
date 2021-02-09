import time
from copy import deepcopy
import numpy.random as npr
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException

class ImproviseHouseBuildingDiscreteEnv(PyBulletEnv):
  ''''''
  def __init__(self, config):
    config['check_random_obj_valid'] = True
    super(ImproviseHouseBuildingDiscreteEnv, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletEnv()
      try:
        self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
        for i in range(self.num_obj-1):
          self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation, z_scale=npr.choice([1, 2], p=[0.75, 0.25]))
      except NoValidPositionException:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    rand_objs = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    if roofs[0].getZPosition() < 1.4*self.min_block_size:
      return False

    rand_obj_combs = combinations(rand_objs, 2)
    for (obj1, obj2) in rand_obj_combs:
      if self._checkOnTop(obj1, roofs[0]) and self._checkOnTop(obj2, roofs[0]):
        return True
    return False

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(ImproviseHouseBuildingDiscreteEnv, self).isSimValid()

def createImproviseHouseBuildingDiscreteEnv(config):
  return ImproviseHouseBuildingDiscreteEnv(config)
