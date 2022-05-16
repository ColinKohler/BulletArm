import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
from itertools import combinations
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

class ImproviseHouseBuildingRandomEnv(BaseEnv):
  ''''''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 5
    if 'max_steps' not in config:
      config['max_steps'] = 10
    config['check_random_obj_valid'] = True
    super(ImproviseHouseBuildingRandomEnv, self).__init__(config)
    self.base_1_objs = []
    self.base_2_objs = []

  def generateBasePair(self):
    scale1 = np.random.uniform(1, 2)
    scale2 = 3-scale1
    return self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation, z_scale=scale1) + \
           self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation, z_scale=scale2)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
        self.base_1_objs = self.generateBasePair()
        self.base_2_objs = self.generateBasePair()
        for i in range(self.num_obj-5):
          self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation, z_scale=np.random.uniform(1, 2))

      except NoValidPositionException:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    rand_objs = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    if roofs[0].getZPosition() < 1.8*self.min_block_size:
      return False
    if not self._checkObjUpright(roofs[0], threshold=np.pi/20):
      return False

    rand_obj_combs = combinations(rand_objs, 2)
    for (obj1, obj2) in rand_obj_combs:
      if self._checkOnTop(obj1, roofs[0]) and self._checkOnTop(obj2, roofs[0]):
        return True
    return False

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(ImproviseHouseBuildingRandomEnv, self).isSimValid()

def createImproviseHouseBuildingRandomEnv(config):
  return ImproviseHouseBuildingRandomEnv(config)
