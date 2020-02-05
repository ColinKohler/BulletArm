import numpy as np
import numpy.random as npr
import pybullet as pb

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.abstract_structure_planner import AbstractStructurePlanner
from helping_hands_rl_envs.simulators import constants

class HouseBuilding2Planner(BasePlanner, AbstractStructurePlanner):
  def __init__(self, env, config):
    super(HouseBuilding2Planner, self).__init__(env, config)
    AbstractStructurePlanner.__init__(self, env)

  def getNextAction(self):
    if self.env._isHolding():
      return self.getPlacingAction()
    else:
      return self.getPickingAction()

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
      return self.pickSecondTallestObjOnTop(objects=roofs, side_grasp=True)

  def getPlacingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))

    assert len(blocks) == 2
    assert len(roofs) == 1

    block1_pos = blocks[0].getPosition()
    block2_pos = blocks[1].getPosition()
    dist = np.linalg.norm(np.array(block1_pos) - np.array(block2_pos))

    valid_block_pos = self.dist_valid(dist)

    if self.env._isObjectHeld(roofs[0]):
      # holding roof, but block pos not valid => place roof on arbitrary pos
      if not valid_block_pos:
        return self.placeOnGround(self.env.max_block_size * 3, self.env.max_block_size * 3)
      # holding roof, block pos valid => place roof on top
      else:
        return self.placeOnTopOfMultiple(blocks)
    # holding block, place block on valid pos
    else:
      if self.env._isObjectHeld(blocks[0]):
        other_block = blocks[1]
      else:
        other_block = blocks[0]
      return self.placeNearAnother(other_block, 1.5*self.env.max_block_size, 1.8*self.env.max_block_size, self.env.max_block_size * 3, self.env.max_block_size * 3)
