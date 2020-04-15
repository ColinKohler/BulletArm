import numpy as np
import numpy.random as npr
import pybullet as pb
from itertools import combinations

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class ImproviseHouseBuilding3DeconstructPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(ImproviseHouseBuilding3DeconstructPlanner, self).__init__(env, config)
    self.objs_to_remove = []

  def getStepLeft(self):
    return 100

  def pickTallestObjOnTop(self, objects=None, side_grasp=False):
    """
    pick up the highest object that is on top
    :param objects: pool of objects
    :param side_grasp: grasp on the side of the object (90 degree), should be true for triangle, brick, etc
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
        if self.env.object_types[obj] in [constants.ROOF, constants.TRIANGLE]:
          side_grasp = True
        break
    if side_grasp:
      r += np.pi / 2
      while r < 0:
        r += np.pi
      while r > np.pi:
        r -= np.pi
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPickingAction(self):
    if self.env.checkStructure():
      self.objs_to_remove = [o for o in self.env.objects]
    if not self.objs_to_remove:
      return self.pickTallestObjOnTop()
    return self.pickTallestObjOnTop(self.objs_to_remove)

  def getPlacingAction(self):
    return self.placeOnGround(self.env.max_block_size * 2, self.env.max_block_size * 3)
