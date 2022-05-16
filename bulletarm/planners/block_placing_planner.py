import numpy as np
import numpy.random as npr

from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class BlockPlacingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BlockPlacingPlanner, self).__init__(env, config)

  def placeOnHighestObj(self, objects=None):
    """
    place on the highest object
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)
    x, y, z, rz = object_poses[0][0], object_poses[0][1], object_poses[0][2]+self.env.place_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if not self.isObjectHeld(obj):
        x, y, z, rz = pose[0], pose[1], pose[2]+self.env.place_offset, pose[5]
        break
    rx = self.env.pick_rx

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, rx))

  def getPickingAction(self):
    return self.pickSecondTallestObjOnTop()

  def getPlacingAction(self):
    return self.placeOnHighestObj()

  def getStepsLeft(self):
    if not self.isSimValid():
      return 100
    return self.getNumTopBlock() - 1
