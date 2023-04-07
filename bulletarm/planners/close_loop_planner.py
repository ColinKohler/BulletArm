import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.base_planner import BasePlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopPlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.dpos = config['dpos'] if 'dpos' in config else 0.05
    self.drot = config['drot'] if 'drot' in config else np.pi / 4

  def getNextActionToCurrentTarget(self, pos_tol=None, rot_tol=None):
    pos_tol = pos_tol if pos_tol else self.dpos
    rot_tol = rot_tol if rot_tol else self.drot

    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < pos_tol) and np.abs(r) < rot_tol:
      self.current_target = None
    return self.env._encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def getActionByGoalPose(self, goal_pos, goal_rot):
    current_pos = self.env.robot._getEndEffectorPosition()
    current_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())
    pos_diff = goal_pos - current_pos
    rot_diff = np.array(goal_rot) - current_rot

    pos_diff[pos_diff // self.dpos > 0] = self.dpos
    pos_diff[pos_diff // -self.dpos > 0] = -self.dpos

    rot_diff[rot_diff // self.drot > 0] = self.drot
    rot_diff[rot_diff // -self.drot > 0] = -self.drot

    x, y, z, r = pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]

    return x, y, z, r
