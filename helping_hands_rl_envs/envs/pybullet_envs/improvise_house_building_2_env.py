from copy import deepcopy
import numpy as np
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException

class ImproviseHouseBuilding2Env(PyBulletEnv):
  ''''''
  def __init__(self, config):
    config['check_random_obj_valid'] = True
    super(ImproviseHouseBuilding2Env, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletEnv()
      try:
        self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
        for i in range(self.num_obj-1):
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
    rand_objs = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    rand_obj_combs = combinations(rand_objs, 2)
    for (obj1, obj2) in rand_obj_combs:
      if self._checkOnTop(obj1, roofs[0]) and self._checkOnTop(obj2, roofs[0]):
        return True
    return False

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(ImproviseHouseBuilding2Env, self).isSimValid()

def createImproviseHouseBuilding2Env(config):
  return ImproviseHouseBuilding2Env(config)
