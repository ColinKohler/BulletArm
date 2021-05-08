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
    self.reset()

  def reset(self):
    self.ready_santilize = False

  def getPickingAction(self):
    if self.env.resetted:
      self.reset()
      self.env.resetted = False

    if self.ready_santilize:
        return self.santilize()

    on_table_obj, on_table_obj_type = self.env.OnTableObj()
    if self.env.placed_swab:
      if on_table_obj == [] or on_table_obj is None:
        return self.santilize()
      return self.pickStickOnTop(on_table_obj)

    if constants.TEST_TUBE in on_table_obj_type:
      # if constants.SWAB in on_table_obj_type:
      #   return self.pickStickOnTop(on_table_obj)
      return self.pickStickOnTop(self.env.getSwabs())
    return self.pickStickOnTop(self.env.getTubes())

  def pickStickOnTop(self, objects=None):
    if objects is None or objects == []: objects = self.env.objects
    objects, object_poses = self.getSizeSortedObjPoses(objects=objects)
    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][5]
    obj_list = [(pair) for pair in zip (objects, object_poses)]
    random.shuffle(obj_list)
    for obj, pose in obj_list:
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2], pose[5]
        r += 1.57
        z -= 0.002
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def santilize(self):
    x, y, z = self.env.santilizing_box_pos
    z = 0.03
    r = 0
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPlacingAction(self):
    if self.env.placed_swab:
      x, y, z = self.env.used_tube_box_pos
      z = 0.02
      r = 1.57
      self.ready_santilize = True
      return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)
    in_hand_obj = self.env.robot.getPickedObj(self.env.objects)
    if in_hand_obj == [] or in_hand_obj is None:
      rand_x, rand_y, rot = 0, 0, 0
      rand_z = 0.1
    elif in_hand_obj.object_type_id == constants.TEST_TUBE:
      rand_x = 0.1 * np.random.rand()
      rand_y = 0.1 * np.random.rand() + 0.1
      rot = 1.57
      rand_z = 0.02
    else:
      rand_x = 0.1 * np.random.rand()
      rand_y = 0.1 * np.random.rand() - 0.1
      rot = 0.
      rand_z = 0.02
    return self.encodeAction(constants.PLACE_PRIMATIVE, 0.5 + rand_x, rand_y, rand_z, rot)

  def getStepsLeft(self):
    return 100