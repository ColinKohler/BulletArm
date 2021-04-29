import numpy as np
from scipy import ndimage
import numpy.random as npr
import matplotlib.pyplot as plt

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

import matplotlib.pyplot as plt

class CovidTestPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(CovidTestPlanner, self).__init__(env, config)
    self.ready_santilize = False
    self.ready_packing_tube = False

  def getPickingAction(self):
    if self.ready_santilize:
        return self.santilize()

    on_table_obj, on_table_obj_type = self.env.OnTableObj()
    if constants.TEST_TUBE in on_table_obj_type:
      if constants.SWAB in on_table_obj_type:
        self.ready_packing_tube = True
        return self.pickLargestObjOnTop(on_table_obj)
      return self.pickShortestObjOnTop(self.env.getSwabs())
    return self.pickLargestObjOnTop(self.env.getTubes())

  def santilize(self):
    x, y, z = self.env.santilizing_box_pos
    z = 0.03
    r = 1.57
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPlacingAction(self):
    if self.ready_packing_tube:
      x, y, z = self.env.old_tube_box_pos
      z = 0.02
      r = 1.57
      self.ready_santilize = True
      return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)
    return self.placeOnGround(min_dist=0.01)

  def getStepsLeft(self):
    return 100