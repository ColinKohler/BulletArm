import numpy as np
import numpy.random as npr
from functools import reduce

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class BrickStackingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BrickStackingPlanner, self).__init__(env, config)

  def getPickingAction(self):
    return self.pickShortestObjOnTop(objects=self.getObjects(obj_type=constants.CUBE))

  def getPlacingAction(self):
    bricks = self.getObjects(obj_type=constants.BRICK)
    return self.placeOn(bricks[0], 3*self.getMaxBlockSize(), len(self.getObjects(obj_type=constants.CUBE)))

  def getStepLeft(self):
    if not self.isSimValid():
      return 100
    step_left = 2*(len(self.getObjects(obj_type=constants.CUBE)) - reduce(lambda x, y: x+y, [len(self.getObjectsOnTopOf(x)) for x in self.getObjects(obj_type=constants.BRICK)]))
    if any([self.isObjectHeld(x) for x in self.getObjects(obj_type=constants.CUBE)]):
      step_left -= 1
    return step_left