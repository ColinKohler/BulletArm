import numpy as np
import numpy.random as npr
import pybullet as pb
from itertools import combinations

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class ImproviseHouseBuilding3DeconstructPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(ImproviseHouseBuilding3DeconstructPlanner, self).__init__(env, config)

  def getStepLeft(self):
    return 100

  def getPickingAction(self):
    rand_objs = list(filter(lambda x: self.env.object_types[x] == constants.RANDOM, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))
    if self.env.checkStructure():
      return self.pickRandomObjOnTop(objects=roofs, side_grasp=True)
    else:
      return self.pickTallestObjOnTop(objects=rand_objs)

  def getPlacingAction(self):
    return self.placeOnGround(self.env.max_block_size * 2, self.env.max_block_size * 3)
