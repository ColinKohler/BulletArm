import numpy as np
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class BottleTrayPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BottleTrayPlanner, self).__init__(env, config)

  def getPickingAction(self):
    # return self.pickSecondTallestObjOnTop(self.env.getObjsOutsideBox())
    objects = self.env.getObjsOutsideBox()
    objects, object_poses = self.getSizeSortedObjPoses(objects=objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2], 0
        break
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPlacingAction(self):
    x, y = self.env.place_pos_candidate[len(self.env.getObjsOutsideBox())-1]
    z = self.env.place_offset
    r = (self.env.box_rz + np.pi/4) % np.pi
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)


  def getStepsLeft(self):
    return 100
