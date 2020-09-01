import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class DrawerOpeningPlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def getNextAction(self):
    handle_pos = self.env.drawer.getHandlePosition()
    drawer_rot = self.env.drawer_rot
    rot = (0, -np.pi/2, drawer_rot)
    m = np.array(transformations.euler_matrix(*rot))[:3, :3]
    pos = handle_pos - m[:, 2] * 0.02

    return self.encodeAction(0, pos[0], pos[1], pos[2], rot)

  def getStepLeft(self):
    return 100