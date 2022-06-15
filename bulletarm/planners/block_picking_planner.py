import numpy as np
import numpy.random as npr

from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class BlockPickingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BlockPickingPlanner, self).__init__(env, config)

  def getNextAction(self):
    return self.pickTallestObjOnTop()
