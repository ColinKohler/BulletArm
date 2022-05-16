import numpy as np
import numpy.random as npr
import pybullet as pb

from bulletarm.planners.block_stacking_planner import BlockStackingPlanner
from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class HouseBuilding2Planner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(HouseBuilding2Planner, self).__init__(env, config)

  def checkFirstLayer(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    block1_pos = blocks[0].getPosition()
    block2_pos = blocks[1].getPosition()
    return block1_pos[-1] < self.getMaxBlockSize() and \
           block2_pos[-1] < self.getMaxBlockSize() and \
           self.getDistance(blocks[0], blocks[1]) < 2.2 * self.getMaxBlockSize()


  def getStepsLeft(self):
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))

    if not self.isSimValid():
      return 100
    if self.checkTermination():
      return 0
    if self.checkFirstLayer():
      step_left = 2
      if self.isObjectHeld(roofs[0]):
        step_left -= 1
    else:
      step_left = 4
      if self.isHolding():
        if self.isObjectHeld(roofs[0]):
          step_left += 1
        else:
          step_left -= 1
    return step_left

  def dist_valid(self, d):
      return 1.5 * self.env.max_block_size < d < 2 * self.env.max_block_size

  def getPickingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))

    assert len(blocks) == 2
    assert len(roofs) == 1

    block1_pos = blocks[0].getPosition()
    block2_pos = blocks[1].getPosition()
    dist = np.linalg.norm(np.array(block1_pos) - np.array(block2_pos))
    valid_block_pos = self.dist_valid(dist)
    # block pos not valid, adjust block pos => pick block
    if not valid_block_pos:
      return self.pickSecondTallestObjOnTop(objects=blocks)
    # block pos valid, pick roof
    else:
      return self.pickSecondTallestObjOnTop(objects=roofs)

  def getPlacingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))

    assert len(blocks) == 2
    assert len(roofs) == 1

    block1_pos = blocks[0].getPosition()
    block2_pos = blocks[1].getPosition()
    dist = np.linalg.norm(np.array(block1_pos) - np.array(block2_pos))

    valid_block_pos = self.dist_valid(dist)

    if self.isObjectHeld(roofs[0]):
      # holding roof, but block pos not valid => place roof on arbitrary pos
      if not valid_block_pos:
        return self.placeOnGround(self.env.max_block_size * 3, self.env.max_block_size * 3)
      # holding roof, block pos valid => place roof on top
      else:
        return self.placeOnTopOfMultiple(blocks)
    # holding block, place block on valid pos
    else:
      if self.isObjectHeld(blocks[0]):
        other_block = blocks[1]
      else:
        other_block = blocks[0]
      return self.placeNearAnother(other_block, 1.7*self.env.max_block_size, 1.8*self.env.max_block_size, self.env.max_block_size * 3, self.env.max_block_size * 3)
