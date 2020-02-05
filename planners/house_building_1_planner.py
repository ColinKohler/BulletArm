import numpy as np
import numpy.random as npr
import pybullet as pb

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.abstract_structure_planner import AbstractStructurePlanner
from helping_hands_rl_envs.simulators import constants

class HouseBuilding1Planner(BasePlanner, AbstractStructurePlanner):
  def __init__(self, env, config):
    super(HouseBuilding1Planner, self).__init__(env, config)
    AbstractStructurePlanner.__init__(self, env)

  def getNextAction(self):
    if self.env._isHolding():
      return self.getPlacingAction()
    else:
      return self.getPickingAction()

  def getPickingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))
    # blocks not stacked, pick block
    if not self.env._checkStack(blocks):
      return self.pickSecondHighestObjOnTop(objects=blocks)
    # blocks stacked, pick triangle
    else:
      return self.pickSecondHighestObjOnTop(objects=triangles, side_grasp=True)

  def getPlacingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))
    # holding triangle, but block not stacked, put down triangle
    if self.env._isObjectHeld(triangles[0]) and not self.env._checkStack(blocks):
      return self.placeOnGround(self.env.max_block_size*3, self.env.max_block_size*3)
    # stack on block
    else:
      return self.placeOnHighestObj(blocks)
