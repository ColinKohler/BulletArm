import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.base_planner import BasePlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopPlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.dpos = config['dpos'] if 'dpos' in config else 0.05
    self.drot = config['drot'] if 'drot' in config else np.pi / 4

  def getActionByGoalPose(self, goal_pos, goal_rot):
    current_pos = self.env.robot._getEndEffectorPosition()
    current_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())
    pos_diff = goal_pos - current_pos
    rot_diff = np.array(goal_rot) - current_rot

    # R = np.array([[np.cos(-current_rot[-1]), -np.sin(-current_rot[-1])],
    #               [np.sin(-current_rot[-1]), np.cos(-current_rot[-1])]])
    # pos_diff[:2] = R.dot(pos_diff[:2])

    pos_diff[pos_diff // self.dpos > 0] = self.dpos
    pos_diff[pos_diff // -self.dpos > 0] = -self.dpos

    rot_diff[rot_diff // self.drot > 0] = self.drot
    rot_diff[rot_diff // -self.drot > 0] = -self.drot

    x, y, z, r = pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]

    return x, y, z, r
