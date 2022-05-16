import numpy as np
import numpy.random as npr

from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class RandomObjPickingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(RandomObjPickingPlanner, self).__init__(env, config)

  def getNextAction(self):
    return self.pickTallestObjOnTop()

  def getStepsLeft(self):
    if not self.env.isSimValid():
      return 100
    step_left = self.env.num_obj - self.env.obj_grasped
    return step_left
