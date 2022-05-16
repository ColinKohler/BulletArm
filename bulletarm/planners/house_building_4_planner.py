from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

from itertools import permutations

import numpy.random as npr
import numpy as np

class HouseBuilding4Planner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(HouseBuilding4Planner, self).__init__(env, config)

  def getObjs(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    bricks = list(filter(lambda x: self.env.object_types[x] == constants.BRICK, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))

    level1_blocks = list()

    perm = permutations(blocks, 2)
    for b1, b2 in perm:
      if not self.isObjOnGround(b1) or not self.isObjOnGround(b2):
        continue
      if self.getDistance(b1, b2) < 2.3 * self.getMaxBlockSize():
        level1_blocks = [b1, b2]
        break

    level2_blocks = list(set(blocks) - set(level1_blocks))

    return level1_blocks, level2_blocks, bricks, roofs

  def checkFirstLayer(self):
    level1_blocks, level2_blocks, bricks, roofs = self.getObjs()
    if len(level1_blocks) < 2:
      return False
    block1_pos = level1_blocks[0].getPosition()
    block2_pos = level1_blocks[1].getPosition()
    return block1_pos[-1] < self.getMaxBlockSize() and \
           block2_pos[-1] < self.getMaxBlockSize() and \
           self.getDistance(level1_blocks[0], level1_blocks[1]) < 2.3 * self.getMaxBlockSize()

  def checkSecondLayer(self):
    level1_blocks, level2_blocks, bricks, roofs = self.getObjs()
    return self.checkOnTopOf(level1_blocks[0], bricks[0]) and \
           self.checkOnTopOf(level1_blocks[1], bricks[0]) and \
           self.checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1])

  def checkThirdLayer(self):
    level1_blocks, level2_blocks, bricks, roofs = self.getObjs()
    return self.checkOnTopOf(bricks[0], level2_blocks[0]) and \
           self.checkOnTopOf(bricks[0], level2_blocks[1]) and \
           self.checkInBetween(bricks[0], level2_blocks[0], level2_blocks[1])

  def checkForthLayer(self):
    level1_blocks, level2_blocks, bricks, roofs = self.getObjs()
    return self.checkOnTopOf(level2_blocks[0], roofs[0]) and \
           self.checkOnTopOf(level2_blocks[1], roofs[0]) and \
           self.checkInBetween(roofs[0], level2_blocks[0], level2_blocks[1])

  def getPickingAction(self):
    level1_blocks, level2_blocks, bricks, roofs = self.getObjs()
    if not self.checkFirstLayer():
      return self.pickSecondTallestObjOnTop(objects=level1_blocks+level2_blocks)
    elif not self.checkSecondLayer():
      return self.pickSecondTallestObjOnTop(objects=bricks)
    elif not self.checkThirdLayer():
      return self.pickSecondTallestObjOnTop(objects=level2_blocks)
    else:
      return self.pickSecondTallestObjOnTop(objects=roofs)

  def getPlacingAction(self):
    level1_blocks, level2_blocks, bricks, roofs = self.getObjs()
    blocks = level1_blocks + level2_blocks
    if not self.checkFirstLayer():
      if self.getHoldingObjType() is constants.CUBE:
        blocks = list(filter(lambda x: not self.isObjectHeld(x), blocks))
        positions = [o.getXYPosition() for o in blocks + bricks + roofs]
        def sum_dist(obj):
          return np.array(list(map(lambda p: np.linalg.norm(np.array(p) - obj.getXYPosition()), positions))).sum()
        other_obj_idx = np.array(list(map(sum_dist, blocks))).argmax()
        other_object = blocks[other_obj_idx]
        return self.placeNearAnother(other_object, self.getMaxBlockSize()*1.7, self.getMaxBlockSize()*1.8, self.getMaxBlockSize()*2, self.getMaxBlockSize()*3)
      else:
        return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)
    elif not self.checkSecondLayer():
      if self.getHoldingObjType() is constants.BRICK:
        return self.placeOnTopOfMultiple(level1_blocks)
      else:
        return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)
    elif not self.checkThirdLayer():
      if self.getHoldingObjType() is constants.CUBE:
        return self.placeOn(bricks[0], 2.8 * self.getMaxBlockSize(), 2)
      else:
        return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)
    else:
      return self.placeOnTopOfMultiple(level2_blocks)

  def getStepsLeft(self):
    if not self.isSimValid():
      return 100
    if self.checkTermination():
      return 0

    level1_blocks, level2_blocks, bricks, roofs = self.getObjs()

    if not self.checkFirstLayer():
      step_left = 10
      if self.isHolding():
        if self.getHoldingObjType() is constants.CUBE:
          step_left -= 1
        else:
          step_left += 1

    elif not self.checkSecondLayer():
      step_left = 8
      if self.isHolding():
        if self.getHoldingObjType() is constants.BRICK:
          step_left -= 1
        else:
          step_left += 1

    elif not self.checkThirdLayer():
      step_left = 6
      if self.checkOnTopOf(bricks[0], level2_blocks[0]) or self.checkOnTopOf(bricks[0], level2_blocks[1]):
        step_left = 4
      if self.isHolding():
        if self.getHoldingObjType() is constants.CUBE:
          step_left -= 1
        else:
          step_left += 1

    else:
      step_left = 2
      if self.isHolding():
        if self.getHoldingObjType() is constants.ROOF:
          step_left -= 1
        else:
          step_left += 1

    return step_left


