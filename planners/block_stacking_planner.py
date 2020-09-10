import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class BlockStackingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BlockStackingPlanner, self).__init__(env, config)

  def getPickingAction(self):
    return self.pickSecondTallestObjOnTop()

  def getPlacingAction(self):
    return self.placeOnHighestObj()

  def getStepLeft(self):
    if not self.isSimValid():
      return 100
    step_left = 2 * (self.getNumTopBlock() - 1)
    if self.isHolding():
      step_left -= 1
    return step_left
