import numpy as np
import numpy.random as npr
import pybullet as pb

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants

class HouseBuilding2Planner(BlockStackingPlanner):
  def __init__(self, env, config):
    super(HouseBuilding2Planner, self).__init__(env, config)

  def getPickingRoofPlan(self, roofs):
    roof_pos, roof_rot = roofs[0].getPose()
    roof_rot = pb.getEulerFromQuaternion(roof_rot)
    x = roof_pos[0]
    y = roof_pos[1]
    z = roof_pos[2] - self.env.pick_offset
    r = -(roof_rot[2] + np.pi / 2)
    while r < 0:
      r += np.pi
    while r > np.pi:
      r -= np.pi
    return self.env._encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPickingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))

    assert len(blocks) == 2
    assert len(roofs) == 1

    block1_pos = blocks[0].getPosition()
    block2_pos = blocks[1].getPosition()
    max_block_size = self.env.max_block_size
    dist = np.linalg.norm(np.array(block1_pos) - np.array(block2_pos))
    def dist_valid(d):
      return 1.5 * max_block_size < d < 2 * max_block_size
    valid_block_pos = dist_valid(dist)
    # block pos not valid, adjust block pos => pick block
    if not valid_block_pos:
      return BlockStackingPlanner.getPickingAction(self)
    # block pos valid, pick roof
    else:
      return self.getPickingRoofPlan(roofs)

  def getPlacingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    roofs = list(filter(lambda x: self.env.object_types[x] == constants.ROOF, self.env.objects))

    assert len(blocks) == 2
    assert len(roofs) == 1

    block1_pos = blocks[0].getPosition()
    block2_pos = blocks[1].getPosition()
    max_block_size = self.env.max_block_size
    dist = np.linalg.norm(np.array(block1_pos) - np.array(block2_pos))

    def dist_valid(d):
      return 1.5 * max_block_size < d < 2 * max_block_size

    valid_block_pos = dist_valid(dist)

    if self.env._isObjectHeld(roofs[0]):
      # holding roof, but block pos not valid => place roof on arbitrary pos
      if not valid_block_pos:
        block_pos = [o.getXYPosition() for o in blocks]
        try:
          place_pos = self.env._getValidPositions(self.env.max_block_size * 3,
                                                  self.env.max_block_size * 2,
                                                  block_pos,
                                                  1)[0]
        except NoValidPositionException:
          place_pos = self.env._getValidPositions(self.env.max_block_size * 3,
                                                  self.env.max_block_size * 2,
                                                  [],
                                                  1)[0]
        x, y, z, r = place_pos[0], place_pos[1], self.env.place_offset, 0
        return self.env._encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)
      # holding roof, block pos valid => place roof on top
      else:
        block_pos = [o.getPosition() for o in blocks]
        middle_point = np.mean((np.array(block_pos[0]), np.array(block_pos[1])), axis=0)
        x, y, z = middle_point[0], middle_point[1], middle_point[2] + self.env.place_offset
        slop = (block_pos[0][1] - block_pos[1][1]) / (block_pos[0][0] - block_pos[1][0])
        r = -np.arctan(slop) - np.pi / 2
        while r < 0:
          r += np.pi
        return self.env._encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)
    # holding block, place block on valid pos
    else:
      place_pos = self.env._getValidPositions(self.env.max_block_size * 2, self.env.max_block_size * 2, [], 1)[0]
      for i in range(10000):
        if self.env._isObjectHeld(blocks[0]):
          other_block = blocks[1]
        else:
          other_block = blocks[0]
        other_block_pos = other_block.getPosition()
        roof_pos = [roofs[0].getXYPosition()]
        try:
          place_pos = self.env._getValidPositions(self.env.max_block_size * 2,
                                                  self.env.max_block_size * 2,
                                                  roof_pos,
                                                  1)[0]
        except NoValidPositionException:
          continue
        dist = np.linalg.norm(np.array(other_block_pos[:-1]) - np.array(place_pos))
        if dist_valid(dist):
          break
      x, y, z, r = place_pos[0], place_pos[1], self.env.place_offset, 0
      return self.env._encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)
