import numpy as np
import numpy.random as npr
import pybullet as pb

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.abstract_structure_planner import AbstractStructurePlanner
from helping_hands_rl_envs.simulators import constants

class HouseBuilding3Planner(BasePlanner, AbstractStructurePlanner):
  def __init__(self, env, config):
    super(HouseBuilding3Planner, self).__init__(env, config)
    AbstractStructurePlanner.__init__(self, env)

    self.blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    self.roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))
    self.bricks = list(filter(lambda x: self.env.object_types[x] == constants.BRICK, self.env.objects))

    assert len(self.blocks) == 2
    assert len(self.roofs) == 1
    assert len(self.bricks) == 1

  def getNextAction(self):
    if self.env._isHolding():
      return self.getPlacingAction()
    else:
      return self.getPickingAction()

  def dist_valid(self, d):
      return 1.5 * self.env.max_block_size < d < 2 * self.env.max_block_size

  def checkFirstLayer(self):
    block1_pos = self.blocks[0].getPosition()
    block2_pos = self.blocks[1].getPosition()
    return block1_pos[-1] < self.getMaxBlockSize() and \
           block2_pos[-1] < self.getMaxBlockSize() and \
           self.getDistance(self.blocks[0], self.blocks[1]) < 2.2 * self.getMaxBlockSize()

  def checkSecondLayer(self):
    return self.checkOnTop(self.blocks[0], self.bricks[0]) and \
           self.checkOnTop(self.blocks[1], self.bricks[0]) and \
           self.checkInBetween(self.bricks[0], self.blocks[0], self.blocks[1])

  def getPickingAction(self):
    # blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    # roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))
    # bricks = list(filter(lambda x: self.env.object_types[x] == constants.BRICK, self.env.objects))
    #
    # assert len(blocks) == 2
    # assert len(roofs) == 1
    # assert len(bricks) == 1
    #
    # block1_pos = blocks[0].getPosition()
    # block2_pos = blocks[1].getPosition()
    # dist = np.linalg.norm(np.array(block1_pos) - np.array(block2_pos))
    # valid_block_pos = self.dist_valid(dist)
    # first layer not done
    if not self.checkFirstLayer():
      # block pos not valid, and roof on brick => pick roof
      if self.checkOnTop(self.bricks[0], self.roofs[0]):
        return self.pickSecondTallestObjOnTop(objects=self.roofs, side_grasp=True)
      # block pos not valid, and brick on top of any block => pick brick
      elif self.checkOnTop(self.blocks[0], self.bricks[0]) or self.checkOnTop(self.blocks[1], self.bricks[0]):
        return self.pickSecondTallestObjOnTop(objects=self.bricks, side_grasp=True)
      # block pos not valid, and roof on top of any block => pick roof
      elif self.checkOnTop(self.blocks[0], self.roofs[0]) or self.checkOnTop(self.blocks[1], self.roofs[0]):
        return self.pickSecondTallestObjOnTop(objects=self.roofs, side_grasp=True)
      # block pos not valid, adjust block pos => pick block
      else:
        return self.pickSecondTallestObjOnTop(objects=self.blocks)

    # first layer done
    else:
      # second layer not done
      if not self.checkSecondLayer():
        # block pos valid, brick is not on top, roof on top of brick => pick roof
        if self.checkOnTop(self.bricks[0], self.roofs[0]):
          return self.pickSecondTallestObjOnTop(objects=self.roofs, side_grasp=True)
        # block pos valid, brick is not on top, and roof on top of any block => pick roof
        elif self.checkOnTop(self.blocks[0], self.roofs[0]) or self.checkOnTop(self.blocks[1], self.roofs[0]):
          return self.pickSecondTallestObjOnTop(objects=self.roofs, side_grasp=True)
        # block pos valid, brick is not on top => pick brick
        else:
          return self.pickSecondTallestObjOnTop(objects=self.bricks, side_grasp=True)

      # second layer done
      else:
        return self.pickSecondTallestObjOnTop(objects=self.roofs, side_grasp=True)

  def getPlacingAction(self):


    # blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    # roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))
    #
    # assert len(blocks) == 2
    # assert len(roofs) == 1
    #
    # block1_pos = blocks[0].getPosition()
    # block2_pos = blocks[1].getPosition()
    # dist = np.linalg.norm(np.array(block1_pos) - np.array(block2_pos))
    #
    # valid_block_pos = self.dist_valid(dist)
    #
    # if self.env._isObjectHeld(roofs[0]):
    #   # holding roof, but block pos not valid => place roof on arbitrary pos
    #   if not valid_block_pos:
    #     return self.placeOnGround(self.env.max_block_size * 3, self.env.max_block_size * 3)
    #   # holding roof, block pos valid => place roof on top
    #   else:
    #     return self.placeOnTopOfMultiple(blocks)
    # # holding block, place block on valid pos
    # else:
    #   if self.env._isObjectHeld(blocks[0]):
    #     other_block = blocks[1]
    #   else:
    #     other_block = blocks[0]
    #   return self.placeNearAnother(other_block, 1.5*self.env.max_block_size, 1.8*self.env.max_block_size, self.env.max_block_size * 3, self.env.max_block_size * 3)
