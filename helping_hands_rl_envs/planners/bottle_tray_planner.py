import numpy as np
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class BottleTrayPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BottleTrayPlanner, self).__init__(env, config)

  def getPickingAction(self):
    # return self.pickSecondTallestObjOnTop(self.env.getObjsOutsideBox())
    return self.pickLargestObjOnTop(self.env.getObjsOutsideBox())

  def getPlacingAction(self):
    x, y = self.env.place_pos_candidate[len(self.env.getObjsOutsideBox())-1]
    z = self.env.place_offset
    r = np.random.random() * np.pi
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)


  def getStepsLeft(self):
    return 100