import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants

class PlayPlanner(BasePlanner):
  def __init__(self, env, config):
    super(PlayPlanner, self).__init__(env, config)

  def getNextAction(self):
    if self.isHolding():
      if npr.rand() < self.rand_place_prob:
        return self.getRandomPlacingAction()
      else:
        return self.getPlayAction()
    else:
      if npr.rand() < self.rand_pick_prob:
        return self.getRandomPickingAction()
      else:
        return self.getPlayAction()

  def getPlayAction(self):
    block_poses = self.env.getObjectPoses()
    pose = block_poses[npr.choice(block_poses.shape[0], 1)][0]

    x, y, z, r = pose[0], pose[1], pose[2], pose[5]
    primative = constants.PLACE_PRIMATIVE if self.env._isHolding() else constants.PICK_PRIMATIVE

    x, y = self.addNoiseToPos(x, y)
    r = self.addNoiseToRot(r)

    return self.env._encodeAction(primative, x, y, z, r)

  # TODO: This is for block stacking so its weird to have this here
  def getStepsLeft(self):
    if not self.isSimValid():
      return 100
    step_left = 2 * (self.getNumTopBlock() - 1)
    if self.isHolding():
      step_left -= 1
    return step_left
