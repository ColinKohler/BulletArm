import numpy as np
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class BoxPalletizingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BoxPalletizingPlanner, self).__init__(env, config)

  def getPickingAction(self):
    # return self.pickSecondTallestObjOnTop(self.env.getObjsOutsideBox())
    return self.pickLargestObjOnTop([self.env.objects[-1]])

  def getPlacingAction(self):
    n_level1, n_level2, n_level3 = self.env.getNEachLevel()
    if n_level1 < 6:
      x, y = self.env.odd_place_pos_candidate[n_level1]
      z = self.env.place_offset
      r = self.env.pallet_rz
    elif n_level2 < 6:
      x, y = self.env.even_place_pos_candidate[n_level2]
      z = self.env.place_offset
      r = self.env.pallet_rz + np.pi / 2
    else:
      x, y = self.env.odd_place_pos_candidate[n_level3]
      z = self.env.place_offset
      r = self.env.pallet_rz
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)


  def getStepsLeft(self):
    return 100