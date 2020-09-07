import numpy as np
import numpy.random as npr
import pybullet as pb

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class HouseBuilding3Planner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(HouseBuilding3Planner, self).__init__(env, config)

  def getStepLeft(self):
    blocks, bricks, roofs = self.getObjs()

    if not self.isSimValid():
      return 100
    if self.checkTermination():
      return 0
    if self.checkFirstLayer():
      if self.checkSecondLayer():
        step_left = 2
        if self.isObjectHeld(roofs[0]):
          step_left = 1
      else:
        step_left = 4
        if self.checkOnTopOf(blocks[0], roofs[0]) or self.checkOnTopOf(blocks[1], roofs[0]):
          step_left += 2
          if self.isObjectHeld(bricks[0]):
            step_left += 1
        elif self.checkOnTopOf(blocks[0], bricks[0]) or self.checkOnTopOf(blocks[1], bricks[0]):
          if self.isObjectHeld(roofs[0]):
            step_left += 1
        elif self.checkOnTopOf(bricks[0], roofs[0]):
          step_left += 2
        else:
          if self.isObjectHeld(bricks[0]):
            step_left -= 1
          elif self.isObjectHeld(roofs[0]):
            step_left += 1
    else:
      step_left = 6
      if self.checkOnTopOf(blocks[0], roofs[0]) or self.checkOnTopOf(blocks[1], roofs[0]):
        step_left += 2
        if self.isObjectHeld(bricks[0]):
          step_left += 1
      elif self.checkOnTopOf(blocks[0], bricks[0]) or self.checkOnTopOf(blocks[1], bricks[0]):
        step_left += 2
        if self.checkOnTopOf(bricks[0], roofs[0]):
          step_left += 2
        elif self.isObjectHeld(roofs[0]):
          step_left += 1
      elif self.checkOnTopOf(bricks[0], roofs[0]):
        step_left += 2
        if self.isObjectHeld(blocks[0]):
          step_left -= 1
      elif self.isHolding():
        if self.isObjectHeld(roofs[0]) or self.isObjectHeld(bricks[0]):
          step_left += 1
        else:
          step_left -= 1
    return step_left

  def getObjs(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    bricks = list(filter(lambda x: self.env.object_types[x] == constants.BRICK, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))
    return blocks, bricks, roofs

  def checkFirstLayer(self):
    blocks, bricks, roofs = self.getObjs()
    block1_pos = blocks[0].getPosition()
    block2_pos = blocks[1].getPosition()
    return block1_pos[-1] < self.getMaxBlockSize() and \
           block2_pos[-1] < self.getMaxBlockSize() and \
           self.getDistance(blocks[0], blocks[1]) < 2.2 * self.getMaxBlockSize()

  def checkSecondLayer(self):
    blocks, bricks, roofs = self.getObjs()
    return self.checkOnTopOf(blocks[0], bricks[0]) and \
           self.checkOnTopOf(blocks[1], bricks[0]) and \
           self.checkInBetween(bricks[0], blocks[0], blocks[1])

  def getPickingAction(self):
    blocks, bricks, roofs = self.getObjs()

    if not self.checkFirstLayer():
      # block pos not valid, and roof on brick => pick roof
      if self.checkOnTopOf(bricks[0], roofs[0]):
        return self.pickSecondTallestObjOnTop(objects=roofs)
      # block pos not valid, and brick on top of any block => pick brick
      elif self.checkOnTopOf(blocks[0], bricks[0]) or self.checkOnTopOf(blocks[1], bricks[0]):
        return self.pickSecondTallestObjOnTop(objects=bricks)
      # block pos not valid, and roof on top of any block => pick roof
      elif self.checkOnTopOf(blocks[0], roofs[0]) or self.checkOnTopOf(blocks[1], roofs[0]):
        return self.pickSecondTallestObjOnTop(objects=roofs)
      # block pos not valid, adjust block pos => pick block
      else:
        return self.pickSecondTallestObjOnTop(objects=blocks)

    # first layer done
    else:
      # second layer not done
      if not self.checkSecondLayer():
        # block pos valid, brick is not on top, roof on top of brick => pick roof
        if self.checkOnTopOf(bricks[0], roofs[0]):
          return self.pickSecondTallestObjOnTop(objects=roofs)
        # block pos valid, brick is not on top, and roof on top of any block => pick roof
        elif self.checkOnTopOf(blocks[0], roofs[0]) or self.checkOnTopOf(blocks[1], roofs[0]):
          return self.pickSecondTallestObjOnTop(objects=roofs)
        # block pos valid, brick is not on top => pick brick
        else:
          return self.pickSecondTallestObjOnTop(objects=bricks)

      # second layer done
      else:
        return self.pickSecondTallestObjOnTop(objects=roofs)

  def getPlacingAction(self):
    blocks, bricks, roofs = self.getObjs()

    # holding brick
    if self.isObjectHeld(bricks[0]):
      # holding brick, but block pos not valid => place brick on arbitrary pos
      if not self.checkFirstLayer():
        return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)
      # holding brick, block pos valid => place brick on top
      else:
        return self.placeOnTopOfMultiple(blocks)

    # holding roof
    elif self.isObjectHeld(roofs[0]):
      # holding roof, but block pos not valid or brick not on top of blocks => place roof on arbitrary pos
      if not (self.checkFirstLayer() and self.checkSecondLayer()):
        return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)
      # holding roof, block and brick pos valid => place roof on top
      else:
        return self.placeOnHighestObj(bricks)
    # holding block, place block on valid pos
    else:
      if self.isObjectHeld(blocks[0]):
        other_block = blocks[1]
      else:
        other_block = blocks[0]
      return self.placeNearAnother(other_block, self.getMaxBlockSize()*1.7, self.getMaxBlockSize()*1.8, self.getMaxBlockSize()*2, self.getMaxBlockSize()*3)
