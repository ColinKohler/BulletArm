import numpy as np
import numpy.random as npr

from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class PlayPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(PlayPlanner, self).__init__(env, config)

  def getNextAction(self):
    if self.isHolding():
      if npr.rand() < self.rand_place_prob:
        return self.getRandomPlacingAction()
      else:
        return self.getPlayAction()
    else:
      if npr.rand() < self.rand_pick_prob:
        return self.getRandomPickingAction()
      else:
        return self.getPlayAction()

  def getPlayAction(self):
    block_poses = self.env.getObjectPoses()
    pose = block_poses[npr.choice(block_poses.shape[0], 1)][0]

    x, y, z, r = pose[0], pose[1], pose[2], pose[5]
    primative = constants.PLACE_PRIMATIVE if self.env._isHolding() else constants.PICK_PRIMATIVE

    return self.encodeAction(primative, x, y, z, r)

  # def getStepsLeft(self):
  #   blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
  #   triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))

  #   if not self.isSimValid():
  #     return 100
  #   if self.checkTermination():
  #     return 0

  #   triangleOnTop = any([self.checkOnTopOf(block, triangles[0]) for block in blocks])
  #   if self.getNumTopBlock(blocks+triangles) > 1 and triangleOnTop:
  #     if any([self.isObjectHeld(block) for block in blocks]):
  #       steps_left = 6
  #     else:
  #       steps_left = 4
  #   else:
  #     steps_left = 0

  #   steps_left += 2 * (self.getNumTopBlock(blocks+triangles) - 1)
  #   if self.isHolding():
  #     steps_left -= 1
  #     if self.isObjectHeld(triangles[0]) and self.getNumTopBlock(blocks+triangles) > 2:
  #       steps_left += 2

  #   return steps_left

  # TODO: This is for block stacking so its weird to have this here
  # def getStepsLeft(self):
  #   if not self.isSimValid():
  #     return 100
  #   step_left = 2 * (self.getNumTopBlock() - 1)
  #   if self.isHolding():
  #     step_left -= 1
  #   return step_left

  def getStepsLeft(self):
    return 0
