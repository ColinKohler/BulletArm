import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants

class BlockPickingPlanner(BasePlanner):
  def __init__(self, env, config):
    super(BlockPickingPlanner, self).__init__(env, config)

  def getNextAction(self):
    block_poses = self.env.getObjectPoses()

    x = block_poses[0][0]
    y = block_poses[0][1]
    z = block_poses[0][2]
    r = block_poses[0][5] # Last rot dim is the only one we care about i.e. (x,y,z)

    if self.pos_noise: x, y = self.addNoiseToPos(x, y)
    if self.rot_noise: rot = self.addNoiseToRot(rot)

    return self.env._encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)
