import numpy as np
import scipy
from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopBlockPullingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.current_target = None
    self.stage = 0

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      self.current_target = None
    return self.env._encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def setNewTarget(self):
    obj1_pos = self.env.objects[0].getPosition()
    obj2_pos = self.env.objects[1].getPosition()
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress([obj1_pos[0], obj2_pos[0]], [obj1_pos[1], obj2_pos[1]])
    if np.isnan(slope):
      slope = np.inf
    d = 0.08
    r = np.arctan(slope)
    g1 = obj1_pos[0] + d * np.cos(r), obj1_pos[1] + d * np.sin(r)
    g2 = obj1_pos[0] - d * np.cos(r), obj1_pos[1] - d * np.sin(r)
    d1 = np.linalg.norm(np.array(g1) - np.array([obj2_pos[0], obj2_pos[1]]))
    d2 = np.linalg.norm(np.array(g2) - np.array([obj2_pos[0], obj2_pos[1]]))
    if d1 > d2:
      g = g1
    else:
      g = g2
    rot = r + np.pi/2
    if rot > np.pi/2:
      rot -= np.pi

    if self.stage == 0:
      self.current_target = (g[0], g[1], 0.1), (0, 0, rot), constants.PLACE_PRIMATIVE
      self.stage = 1
    elif self.stage == 1:
      self.current_target = (g[0], g[1], 0.03), (0, 0, rot), constants.PLACE_PRIMATIVE
      self.stage = 2
    else:
      self.current_target = (obj2_pos[0], obj2_pos[1], 0.03), (0, 0, rot), constants.PLACE_PRIMATIVE
      self.stage = 0

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.current_target = None
      self.stage = 0
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100
