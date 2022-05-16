import numpy as np
import numpy.random as npr
import pybullet as pb
from itertools import combinations

from bulletarm.planners.block_stacking_planner import BlockStackingPlanner
from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants


class DeconstructPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(DeconstructPlanner, self).__init__(env, config)
    self.objs_to_remove = []

  def getStepsLeft(self):
    return 100

  def pickTallestObjOnTop(self, objects=None):
    """
    pick up the highest object that is on top
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2]+self.env.pick_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2]+self.env.pick_offset, pose[5]
        if obj in self.objs_to_remove:
          self.objs_to_remove.remove(obj)
        break
      while r < 0:
        r += np.pi
      while r > np.pi:
        r -= np.pi
    if not self.random_orientation:
      r = 0
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPickingAction(self):
    if self.env.checkStructure():
      self.objs_to_remove = [o for o in self.env.structure_objs]
    if not self.objs_to_remove:
      return self.pickTallestObjOnTop()
    return self.pickTallestObjOnTop(self.objs_to_remove)

  def getPlacingAction(self):
    return self.placeOnGround(self.env.max_block_size * 2, self.env.max_block_size * 2.7)
