import numpy as np
import scipy
from bulletarm.planners.close_loop_planner import CloseLoopPlanner

class CloseLoopBlockPushingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.current_target = None
    self.stage = 0
    self.pre_push_start_pos = self.env.workspace.mean(1)
    self.push_start_pos = self.env.workspace.mean(1)
    self.push_end_pos = self.env.workspace.mean(1)
    self.push_rot = 0

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    p = self.current_target[2]
    if np.all(np.abs([x, y, z]) < self.dpos) and (not self.random_orientation or np.abs(r) < self.drot):
      self.current_target = None
    return self.env._encodeAction(p, x, y, z, r)

  def setWaypoints(self):
    obj_pos = self.env.getObjectPositions(omit_hold=False)[:, :2].tolist()[0]
    goal_pos = self.env.goal_pos
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress([obj_pos[0], goal_pos[0]],
                                                                         [obj_pos[1], goal_pos[1]])
    if np.isnan(slope):
      slope = np.inf
    d = 0.08
    r = np.arctan(slope)
    p1 = [obj_pos[0] + d * np.cos(r), obj_pos[1] + d * np.sin(r)]
    g1 = [goal_pos[0] + 0.035 * np.cos(r), goal_pos[1] + 0.035 * np.sin(r)]
    p2 = [obj_pos[0] - d * np.cos(r), obj_pos[1] - d * np.sin(r)]
    g2 = [goal_pos[0] - 0.035 * np.cos(r), goal_pos[1] - 0.035 * np.sin(r)]
    d1 = np.linalg.norm(np.array(p1) - np.array([goal_pos[0], goal_pos[1]]))
    d2 = np.linalg.norm(np.array(p2) - np.array([goal_pos[0], goal_pos[1]]))
    if d1 > d2:
      p = p1
      g = g1
    else:
      p = p2
      g = g2
    rot = r + np.pi / 2
    if rot > np.pi / 2:
      rot -= np.pi
    p[0] = np.clip(p[0], self.env.workspace[0][0], self.env.workspace[0][1])
    p[1] = np.clip(p[1], self.env.workspace[1][0], self.env.workspace[1][1])
    self.pre_push_start_pos = (p[0], p[1], self.env.workspace[2][0] + 0.1)
    self.push_start_pos = (p[0], p[1], self.env.workspace[2][0]+0.02)
    self.push_end_pos = (g[0], g[1], self.env.workspace[2][0]+0.02)
    self.push_rot = rot

  def setNewTarget(self):
    if self.stage == 0:
      self.setWaypoints()
      # to pre push start pos
      self.current_target = (self.pre_push_start_pos, self.push_rot, 0.5, 0.5)
      self.stage = 1
    elif self.stage == 1:
      # to push start pos
      self.current_target = (self.push_start_pos, self.push_rot, 0.5, 0.5)
      self.stage = 2
    elif self.stage == 2:
      # to push end pos
      self.current_target = (self.push_end_pos, self.push_rot, 0.5, 0.5)
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