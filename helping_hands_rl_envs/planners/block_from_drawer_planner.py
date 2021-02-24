import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class BlockFromDrawerPlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def getNextAction(self):
    if not self.env.drawer.isDrawerOpen():
      handle_pos = self.env.drawer.getHandlePosition()
      rot = (0, -np.pi / 2, 0)
      return self.encodeAction(constants.PULL_PRIMATIVE, handle_pos[0], handle_pos[1], handle_pos[2], rot)
    elif self.env._isObjectWithinWorkspace(self.env.objects[0]):
      pos = self.env.objects[0].getPosition()
      rot = (0, 0, 0)
      return self.encodeAction(constants.PICK_PRIMATIVE, pos[0], pos[1], pos[2], rot)
    elif self.isHolding():
      pos = [0.1, 0, 0.02]
      rot = (0, 0, 0)
      return self.encodeAction(constants.PLACE_PRIMATIVE, pos[0], pos[1], pos[2], rot)

    else:
      handle_pos = self.env.drawer2.getHandlePosition()
      rot = (0, -np.pi / 2, 0)
      return self.encodeAction(constants.PULL_PRIMATIVE, handle_pos[0], handle_pos[1], handle_pos[2], rot)

  def getStepLeft(self):
    return 100