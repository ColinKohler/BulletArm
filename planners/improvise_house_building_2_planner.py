import numpy as np
import numpy.random as npr
import pybullet as pb
from itertools import combinations

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class ImproviseHouseBuilding2Planner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(ImproviseHouseBuilding2Planner, self).__init__(env, config)

  def getFirstLayerObjs(self):
    rand_objs = list(filter(lambda x: self.env.object_types[x] == constants.RANDOM, self.env.objects))
    rand_obj_combs = combinations(rand_objs, 2)

    for (obj1, obj2) in rand_obj_combs:
      pos1 = obj1.getPosition()
      pos2 = obj2.getPosition()
      if pos1[-1] < self.getMaxBlockSize() and \
          pos2[-1] < self.getMaxBlockSize() and \
          self.getDistance(obj1, obj2) < 3 * self.getMaxBlockSize():
        return obj1, obj2
    return None, None

  def getStepLeft(self):
    return 100

  def dist_valid(self, d):
      return 1.5 * self.env.max_block_size < d < 2 * self.env.max_block_size

  def getPickingAction(self):
    rand_objs = list(filter(lambda x: self.env.object_types[x] == constants.RANDOM, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))
    obj1, obj2 = self.getFirstLayerObjs()
    if obj1 is None:
      return self.pickRandomObjOnTop(objects=rand_objs)
    else:
      return self.pickRandomObjOnTop(objects=roofs)

  def getPlacingAction(self):
    rand_objs = list(filter(lambda x: self.env.object_types[x] == constants.RANDOM and not self.isObjectHeld(x), self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))
    obj1, obj2 = self.getFirstLayerObjs()

    if self.isObjectHeld(roofs[0]):
      if obj1 is None:
        return self.placeOnGround(self.env.max_block_size * 3, self.env.max_block_size * 3)
      else:
        return self.placeOnTopOfMultiple([obj1, obj2])
    else:
      npr.shuffle(rand_objs)
      other_obj = rand_objs[0]
      return self.placeNearAnother(other_obj, 1.7*self.env.max_block_size, 2.8*self.env.max_block_size, self.env.max_block_size * 3, self.env.max_block_size * 3)
