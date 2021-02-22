import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class BlockAdjacentPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(BlockAdjacentPlanner, self).__init__(env, config)

  def getPickingAction(self):
    objects = self.getObjects()
    if self.isAdjacent([objects[0], objects[1]]):
      obj_to_pick = objects[2]
    elif self.isAdjacent([objects[0], objects[2]]):
      obj_to_pick = objects[1]
    elif self.isAdjacent([objects[1], objects[2]]):
      obj_to_pick = objects[0]
    else:
      obj_to_pick = npr.choice(objects)

    pose = self.env.getObjectPoses([obj_to_pick])
    return self.encodeAction(constants.PICK_PRIMATIVE, pose[0][0], pose[0][1], pose[0][2], pose[0][-1])

  def getPlacingAction(self):
    return self.placeAdjacent(npr.choice(self.getObjects()),
                              self.getMaxBlockSize()*1.0,
                              self.getMaxBlockSize()*1.5,
                              self.getMaxBlockSize()*2,
                              self.getMaxBlockSize()*2)

  def getStepsLeft(self):
    if not self.isSimValid():
      return 100
    if self.checkTermination():
      return 0

    objects = self.getObjects()
    # No objects in hand
    if len(objects) == 3:
      if self.isAdjacent(objects):
        return 2
      else:
        return 4
    # Object in hand
    if len(objects) == 2:
      if self.isAdjacent(objects):
        return 1
      else:
        return 3
