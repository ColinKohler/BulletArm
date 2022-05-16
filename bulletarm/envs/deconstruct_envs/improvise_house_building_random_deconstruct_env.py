import time
from copy import deepcopy
import numpy.random as npr
import numpy as np
import pybullet as pb
from itertools import combinations
from bulletarm.envs.deconstruct_envs.deconstruct_env import DeconstructEnv
from bulletarm.pybullet.utils import constants

class ImproviseHouseBuildingRandomDeconstructEnv(DeconstructEnv):
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
    super(ImproviseHouseBuildingRandomDeconstructEnv, self).__init__(config)
    self.terminate_min_dist = 2.7*self.min_block_size

  def checkStructure(self):
    rand_objs = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    if roofs[0].getZPosition() < 1.8 * self.min_block_size:
      return False
    if not self._checkObjUpright(roofs[0], threshold=np.pi / 20):
      return False

    rand_obj_combs = combinations(rand_objs, 2)
    for (obj1, obj2) in rand_obj_combs:
      if self._checkOnTop(obj1, roofs[0]) and self._checkOnTop(obj2, roofs[0]):
        return True
    return False

  def generateStructure(self):
    roof_z = 0.06

    padding = self.max_block_size * 1.5
    min_dist = 1.7 * self.max_block_size
    max_dist = 2.4 * self.max_block_size
    pos1, pos2 = self.get2BaseXY(padding, min_dist, max_dist)

    base1_scale1 = np.random.uniform(1, 2)
    base1_scale2 = 3 - base1_scale1
    base2_scale1 = np.random.uniform(1, 2)
    base2_scale2 = 3 - base2_scale1

    self.generateStructureRandomShapeWithZScale([pos1[0], pos1[1], base1_scale1 * 0.007],
                                                self._getValidOrientation(self.random_orientation), base1_scale1)
    self.generateStructureRandomShapeWithZScale([pos1[0], pos1[1], base1_scale1 * 0.014 + base1_scale2 * 0.007],
                                                self._getValidOrientation(self.random_orientation), base1_scale2)

    self.generateStructureRandomShapeWithZScale([pos2[0], pos2[1], base2_scale1 * 0.007],
                                                self._getValidOrientation(self.random_orientation), base2_scale1)
    self.generateStructureRandomShapeWithZScale([pos2[0], pos2[1], base2_scale1 * 0.014 + base2_scale2 * 0.007],
                                                self._getValidOrientation(self.random_orientation), base2_scale2)

    x, y, r = self.getXYRFrom2BasePos(pos1, pos2)

    self.generateStructureShape((x, y, roof_z), pb.getQuaternionFromEuler([0., 0., r]), constants.ROOF)
    self.wait(50)

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and DeconstructEnv.isSimValid(self)

def createImproviseHouseBuildingRandomDeconstructEnv(config):
  return ImproviseHouseBuildingRandomDeconstructEnv(config)
