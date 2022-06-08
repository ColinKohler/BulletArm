from copy import deepcopy
import numpy as np
from itertools import combinations
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

class ImproviseHouseBuilding2Env(BaseEnv):
  '''Open loop improvise house building 2 task.

  The robot needs to first place two blocks adjacent to each other, then place a roof on top.
  The two base blocks are randomly generated shapes.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    config['check_random_obj_valid'] = True
    super(ImproviseHouseBuilding2Env, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
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
