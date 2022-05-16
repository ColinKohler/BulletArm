import numpy as np
import numpy.random as npr

from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations

class PyramidStackingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(PyramidStackingPlanner, self).__init__(env, config)

  def getPickingAction(self):
    objects = self.getObjects()
    # TODO: This could be cleaner
    if self.isNear(objects[0], objects[1]):
      obj_to_pick = objects[2]
    elif self.isNear(objects[0], objects[2]):
      obj_to_pick = objects[1]
    elif self.isNear(objects[1], objects[2]):
      obj_to_pick = objects[0]
    else:
      obj_to_pick = npr.choice(objects)

    pos, rot = obj_to_pick.getPose()
    rx, ry, rz = transformations.euler_from_quaternion(rot)
    return self.encodeAction(constants.PICK_PRIMATIVE, pos[0], pos[1], pos[2], rz)

  def getPlacingAction(self):
    objects = self.getObjects()

    if self.isNear(objects[0], objects[1]):
      return self.placeOnTopOfMultiple(objects)
    else:
      return self.placeNearAnother(npr.choice(objects),
                                   self.getMaxBlockSize()*1.4,
                                   self.getMaxBlockSize()*1.5,
                                   self.getMaxBlockSize()*2,
                                   self.getMaxBlockSize()*3)

  def getStepsLeft(self):
    if not self.isSimValid():
      return 100
    if self.checkTermination():
      return 0

    objects = self.getObjects()
    # No objects in hand
    if len(objects) == 3:
      if self.isNear(objects[0], objects[1]) or \
         self.isNear(objects[1], objects[2]) or \
         self.isNear(objects[0], objects[2]):
        return 2
      else:
        return 4
    # Object in hand
    if len(objects) == 2:
      if self.isNear(objects[0], objects[1]):
        return 1
      else:
        return 3
