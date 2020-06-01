import numpy as np
import numpy.random as npr
import pybullet as pb

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class HouseBuilding1Planner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(HouseBuilding1Planner, self).__init__(env, config)

  def getStepLeft(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))

    if not self.isSimValid():
      return 100
    step_left = 2 * (self.getNumTopBlock(blocks+triangles) - 1)
    if self.isHolding():
      step_left -= 1
      if self.isObjectHeld(triangles[0]) and self.getNumTopBlock(blocks+triangles) > 2:
        step_left += 2
    return step_left

  def getPickingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))
    # blocks not stacked, pick block
    if not self.checkStack(blocks):
      return self.pickSecondTallestObjOnTop(objects=blocks)
    # blocks stacked, pick triangle
    else:
      return self.pickSecondTallestObjOnTop(objects=triangles)

  def getPlacingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))
    # holding triangle, but block not stacked, put down triangle
    if self.isObjectHeld(triangles[0]) and not self.checkStack(blocks):
      return self.placeOnGround(self.env.max_block_size*3, self.env.max_block_size*3)
    # stack on block
    else:
      return self.placeOnHighestObj(blocks)
