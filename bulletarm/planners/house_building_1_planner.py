import numpy as np
import numpy.random as npr
import pybullet as pb

from bulletarm.planners.block_stacking_planner import BlockStackingPlanner
from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class HouseBuilding1Planner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(HouseBuilding1Planner, self).__init__(env, config)

  def getStepsLeft(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))

    if not self.isSimValid():
      return 100
    if self.checkTermination():
      return 0

    triangleOnTop = any([self.checkOnTopOf(block, triangles[0]) for block in blocks])
    if self.getNumTopBlock(blocks+triangles) > 1 and triangleOnTop:
      if any([self.isObjectHeld(block) for block in blocks]):
        steps_left = 6
      else:
        steps_left = 4
    else:
      steps_left = 0

    steps_left += 2 * (self.getNumTopBlock(blocks+triangles) - 1)
    if self.isHolding():
      steps_left -= 1
      if self.isObjectHeld(triangles[0]) and self.getNumTopBlock(blocks+triangles) > 2:
        steps_left += 2

    return steps_left

  def getPickingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))
    # blocks not stacked, pick block
    if len(blocks) > 1 and not self.checkStack(blocks):
      return self.pickSecondTallestObjOnTop(objects=blocks)
    # blocks stacked, pick triangle
    else:
      return self.pickSecondTallestObjOnTop(objects=triangles)

  def getPlacingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))
    # holding triangle, but block not stacked, put down triangle
    if (self.isObjectHeld(triangles[0]) and not self.checkStack(blocks)) or (len(blocks) == 1 and self.isObjectHeld(blocks[0])):
      return self.placeOnGround(self.env.max_block_size*3, self.env.max_block_size*3)
    # stack on block
    else:
      return self.placeOnHighestObj(blocks)
