import numpy as np
from scipy import ndimage
import numpy.random as npr
import matplotlib.pyplot as plt
import random

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
        return self.pickStickOnTop(on_table_obj)
      return self.pickStickOnTop(self.env.getSwabs())
    return self.pickStickOnTop(self.env.getSwabs())

  def pickStickOnTop(self, objects=None):
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSizeSortedObjPoses(objects=objects)
    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][5]
    obj_list = [(pair) for pair in zip (objects, object_poses)]
    random.shuffle(obj_list)
    for obj, pose in obj_list:
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2], pose[5]
        r += 1.57
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def santilize(self):
    x, y, z = self.env.santilizing_box_pos
    z = 0.03
    r = 0
    self.ready_santilize = False
    self.ready_packing_tube = False
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPlacingAction(self):
    if self.ready_packing_tube:
      x, y, z = self.env.used_tube_box_pos
      z = 0.02
      r = 1.57
      self.ready_santilize = True
      return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)
    return self.encodeAction(constants.PLACE_PRIMATIVE, 0.8, 0, 0.02, 0)

  def getStepsLeft(self):
    return 100