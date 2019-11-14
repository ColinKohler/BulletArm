import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants

class BlockStackingPlanner(BasePlanner):
  def __init__(self, env, config):
    super(BlockStackingPlanner, self).__init__(env, config)

  def getNextAction(self):
    if self.env._isHolding():
      return self.getPlacingAction()
    else:
      return self.getPickingAction()

  def getPickingAction(self):
    blocks, block_poses = self.getBlockPoses(roll=True)

    x, y, z, r = block_poses[0][0], block_poses[0][1], block_poses[0][2], block_poses[0][5]
    for block, pose in zip(blocks, block_poses):
      # TODO: This function could use a better name
      if self.env._isObjOnTop(block):
        x, y, z, r = pose[0], pose[1], pose[2], pose[5]
        break

    if self.pos_noise: x, y = self.addNoiseToPos(x, y)
    if self.rot_noise: rot = self.addNoiseToRot(rot)

    return self.env._encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPlacingAction(self):
    blocks, block_poses = self.getBlockPoses()

    for block, pose in zip(blocks, block_poses):
      if not self.env._isObjectHeld(block):
        x, y, z, r = pose[0], pose[1], pose[2], pose[5]

    if self.pos_noise: x, y = self.addNoiseToPos(x, y)
    if self.rot_noise: rot = self.addNoiseToRot(rot)

    return self.env._encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def getBlockPoses(self, roll=False):
    blocks = np.array(self.env.getObjects())
    block_poses = self.env.getObjectPoses()

    # Sort by block size
    sorted_inds = np.flip(np.argsort(block_poses[:,2], axis=0))

    # TODO: Should get a better var name for this
    if roll:
      sorted_inds = np.roll(sorted_inds, -1)

    blocks = blocks[sorted_inds]
    block_poses = block_poses[sorted_inds]
    return blocks, block_poses
