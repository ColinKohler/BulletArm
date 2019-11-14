import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants

class PlayPlanner(BasePlanner):
  def __init__(self, env, config):
    super(PlayPlanner, self).__init__(env, config)

  def getNextAction(self):
    block_poses = self.env.getObjectPoses()
    pose = block_poses[npr.choice(block_poses.shape[0], 1)][0]

    x, y, z, r = pose[0], pose[1], pose[2], pose[5]
    primative = constants.PLACE_PRIMATIVE if self.env._isHolding() else constants.PICK_PRIMATIVE

    if self.pos_noise: x, y = self.addNoiseToPos(x, y)
    if self.rot_noise: rot = self.addNoiseToRot(rot)

    return self.env._encodeAction(primative, x, y, z, r)
