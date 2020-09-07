import numpy as np
import numpy.random as npr
import pybullet as pb
from itertools import combinations

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class ImproviseHouseBuilding3Planner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(ImproviseHouseBuilding3Planner, self).__init__(env, config)

  def getFirstLayerObjs(self):
    rand_objs = list(filter(lambda x: self.env.object_types[x] == constants.RANDOM, self.env.objects))
    rand_objs_1 = list(filter(lambda x: x.z_scale == 1, rand_objs))
    rand_objs_2 = list(filter(lambda x: x.z_scale == 2, rand_objs))

    rand_obj_1_combs = combinations(rand_objs_1, 2)
    rand_obj_2_combs = combinations(rand_objs_2, 2)

    for (obj1, obj2) in rand_obj_2_combs:
      if self.isObjOnTop(obj1) and self.isObjOnTop(obj2) and self.getDistance(obj1, obj2) < 3.2 * self.getMaxBlockSize():
        return obj1, obj2

    for (obj1, obj2) in rand_obj_1_combs:
      pos1 = obj1.getPosition()
      pos2 = obj2.getPosition()
      if pos1[-1] > 0.5*self.getMaxBlockSize() and pos2[-1] > 0.5*self.getMaxBlockSize() and self.isObjOnTop(obj1) and self.isObjOnTop(obj2)\
          and self.getDistance(obj1, obj2) < 3.2 * self.getMaxBlockSize():
        return obj1, obj2

    for obj1 in rand_objs_1:
      for obj2 in rand_objs_2:
        pos1 = obj1.getPosition()
        if pos1[-1] > 0.5*self.getMaxBlockSize() and self.isObjOnTop(obj1) and self.isObjOnTop(obj2) \
            and self.getDistance(obj1, obj2) < 3.2 * self.getMaxBlockSize():
          return obj1, obj2

    return None, None

  def getIncompleteFirstLayerObjs(self):
    rand_objs = list(filter(lambda x: self.env.object_types[x] == constants.RANDOM, self.env.objects))
    rand_objs_1 = list(filter(lambda x: x.z_scale == 1, rand_objs))
    rand_objs_2 = list(filter(lambda x: x.z_scale == 2, rand_objs))
    for obj1 in rand_objs_1:
      for obj2 in rand_objs_2:
        pos1 = obj1.getPosition()
        if pos1[-1] < 0.5*self.getMaxBlockSize() and self.isObjOnTop(obj1) and self.isObjOnTop(obj2) \
            and self.getDistance(obj1, obj2) < 3.2 * self.getMaxBlockSize():
          return obj1, obj2

    rand_obj_1_combs = combinations(rand_objs_1, 2)
    for (obj1, obj2) in rand_obj_1_combs:
      pos1 = obj1.getPosition()
      pos2 = obj2.getPosition()
      if self.getDistance(obj1, obj2) < 3.2 * self.getMaxBlockSize() and self.isObjOnTop(obj1) and self.isObjOnTop(obj2):
        if pos2[-1] < 0.5*self.getMaxBlockSize():
          return obj2, obj1
        else:
          return obj1, obj2

    return None, None

  def getStepLeft(self):
    return 100

  def getPickingAction(self):
    rand_objs = list(filter(lambda x: self.env.object_types[x] == constants.RANDOM, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))
    obj1, obj2 = self.getFirstLayerObjs()
    if obj1 is None:
      obj1, obj2 = self.getIncompleteFirstLayerObjs()
      rand_objs = list(filter(lambda x: x not in (obj1, obj2), rand_objs))
      return self.pickRandomObjOnTop(objects=rand_objs)
    else:
      return self.pickRandomObjOnTop(objects=roofs)

  def getPlacingAction(self):
    rand_objs = list(filter(lambda x: self.env.object_types[x] == constants.RANDOM and not self.isObjectHeld(x), self.env.objects))
    rand_objs_1 = list(filter(lambda x: x.z_scale == 1, rand_objs))
    rand_objs_2 = list(filter(lambda x: x.z_scale == 2, rand_objs))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))
    obj1, obj2 = self.getFirstLayerObjs()

    if self.isObjectHeld(roofs[0]):
      if obj1 is None:
        return self.placeOnGround(self.env.max_block_size * 3, self.env.max_block_size * 3)
      else:
        return self.placeOnTopOfMultiple([obj1, obj2])
    else:
      if self.getHoldingObj().z_scale == 1:
        if len(rand_objs_1) == 0:
          return self.placeOnGround(self.env.max_block_size * 3, self.env.max_block_size * 3)
        obj1, obj2 = self.getIncompleteFirstLayerObjs()
        if obj1 is not None:
          return self.placeOnHighestObj([obj1])

      npr.shuffle(rand_objs)
      other_obj = rand_objs[0]
      return self.placeNearAnother(other_obj, 1.7*self.env.max_block_size, 2.8*self.env.max_block_size, self.env.max_block_size * 3, self.env.max_block_size * 3)
