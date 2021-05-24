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
    self.place_on = None
    self.prev_place = None

  def getPickingAction(self):
    if self.env.resetted:
      self.reset()
      self.env.resetted = False

    # prepare multiple swab-tube pairs
    in_box_swabs = self.env.getSwabs()
    in_box_tubes = self.env.getTubes()
    on_table_obj, on_table_obj_type = self.env.OnTableObj()
    if len(in_box_swabs) > len(in_box_tubes):
      self.place_on = 'table'
      return self.pickStickOnTop(in_box_swabs)
    elif len(on_table_obj) != 0:
      self.place_on = 'used_tube_box'
      return self.pickStickOnTop(on_table_obj)
    elif len(in_box_tubes) != 0:
      self.place_on = 'table'
      return self.pickStickOnTop(in_box_tubes)
    else:
      return self.santilize()

    # # prepare one swab-tube pair
    # if self.ready_santilize:
    #     return self.santilize()
    # on_table_obj, on_table_obj_type = self.env.OnTableObj()
    # if self.env.placed_swab:
    #   if on_table_obj == [] or on_table_obj is None:
    #     return self.santilize()
    #   return self.pickStickOnTop(on_table_obj)
    #
    # if constants.TEST_TUBE in on_table_obj_type:
    #   # if constants.SWAB in on_table_obj_type:
    #   #   return self.pickStickOnTop(on_table_obj)
    #   return self.pickStickOnTop(self.env.getSwabs())
    # return self.pickStickOnTop(self.env.getTubes())

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
        # z -= 0.001
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def santilize(self):
    x, y, z = self.env.santilizing_box_pos
    z = 0.03
    if self.env.rot_n % 2 == 1:
      r = np.pi / 2
    else:
      r = 0
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPlacingAction(self):
    # for multiple swab-tube pair
    if self.place_on == 'used_tube_box':
      x, y, z = self.env.used_tube_box_pos
      x += 0.03 * np.random.rand() - 0.015
      y += 0.03 * np.random.rand() - 0.015
      z = 0.05
      r = 1.57 + self.env.rot90x + 0.2 * np.random.rand() - 0.1
      self.ready_santilize = True
      return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

    # # for one swab-tube pair
    # if self.env.placed_swab:
    #   x, y, z = self.env.used_tube_box_pos
    #   z = 0.02
    #   r = 1.57
    #   self.ready_santilize = True
    #   return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)
    in_hand_obj = self.env.robot.getPickedObj(self.env.objects)
    if in_hand_obj == [] or in_hand_obj is None:
      rand_x, rand_y, rot = 0, 0, 0
      rand_z = 0.1
    elif in_hand_obj.object_type_id == constants.TEST_TUBE:
      x, y, z = self.env.test_box_pos
      rand_x = x + 0.05 * np.random.rand() - 0.025
      rand_y = y + 0.05 * np.random.rand() - 0.025
      rot = 1.57 + self.env.rot90x
      rand_z = 0.02
      self.prev_place = [rand_x, rand_y, rand_z, rot]
    else:  # placing swab
      rand_x, rand_y, rand_z, rot = self.prev_place
      x, y, z = self.env.test_box_pos
      if self.env.rot_n % 2 == 1:
        rand_x = rand_x + 0.1 * (np.random.rand() > 0.5 - 0.5)
        rand_y = rand_y + 0.05 * np.random.rand() - 0.025
      else:
        rand_x = rand_x + 0.05 * np.random.rand() - 0.025
        rand_y = rand_y + 0.1 * (np.random.rand() > 0.5 - 0.5)
      rot = self.env.rot90x
      rand_z = 0.02
    return self.encodeAction(constants.PLACE_PRIMATIVE, rand_x, rand_y, rand_z, rot)

  def getStepsLeft(self):
    return 100