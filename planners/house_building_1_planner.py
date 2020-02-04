import numpy as np
import numpy.random as npr
import pybullet as pb

from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants

class HouseBuilding1Planner(BlockStackingPlanner):
  def __init__(self, env, config):
    super(HouseBuilding1Planner, self).__init__(env, config)

  def getPickingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))
    # blocks not stacked, pick block
    if not self.env._checkStack(blocks):
      return BlockStackingPlanner.getPickingAction(self)
    # blocks stacked, pick triangle
    else:
      triangle_pos, triangle_rot = triangles[0].getPose()
      # TODO: this should be changed to not dependent on pb
      triangle_rot = pb.getEulerFromQuaternion(triangle_rot)
      x = triangle_pos[0]
      y = triangle_pos[1]
      z = triangle_pos[2] - self.env.pick_offset
      r = -(triangle_rot[2] + np.pi / 2)
      while r < 0:
        r += np.pi
      while r > np.pi:
        r -= np.pi
      return self.env._encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPlacingAction(self):
    blocks = list(filter(lambda x: self.env.object_types[x] == constants.CUBE, self.env.objects))
    triangles = list(filter(lambda x: self.env.object_types[x] == constants.TRIANGLE, self.env.objects))
    # holding triangle, but block not stacked, put down triangle
    if self.env._isObjectHeld(triangles[0]) and not self.env._checkStack(blocks):
      block_pos = [o.getXYPosition() for o in blocks]
      place_pos = self.env._getValidPositions(self.env.block_scale_range[1] * self.env.block_original_size,
                                          self.env.block_scale_range[1] * self.env.block_original_size,
                                          block_pos,
                                          1)[0]
      x = place_pos[0]
      y = place_pos[1]
      z = self.env.place_offset
      r = 0
      return self.env._encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)
    # stack on block
    else:
      return BlockStackingPlanner.getPlacingAction(self)
