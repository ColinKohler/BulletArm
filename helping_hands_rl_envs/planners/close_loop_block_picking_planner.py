import numpy as np

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class CloseLoopBlockPickingPlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.dpos = config['dpos'] if 'dpos' in config else 0.005
    self.drot = config['drot'] if 'drot' in config else np.pi/16

  def getNextAction(self):
    current_pos = self.env.robot._getEndEffectorPosition()
    current_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())

    if not self.env._isHolding():
      block_pos = self.env.objects[0].getPosition()
      block_rot = transformations.euler_from_quaternion(self.env.objects[0].getRotation())

      pos_diff = block_pos - current_pos
      rot_diff = np.array(block_rot) - current_rot

      R = np.array([[np.cos(-current_rot[-1]), -np.sin(-current_rot[-1])],
                    [np.sin(-current_rot[-1]), np.cos(-current_rot[-1])]])
      pos_diff[:2] = R.dot(pos_diff[:2])

      pos_diff[pos_diff // self.dpos > 0] = self.dpos
      pos_diff[pos_diff // -self.dpos > 0] = -self.dpos
      pos_diff[np.abs(pos_diff) < self.dpos] = 0

      rot_diff[rot_diff // self.drot > 0] = self.drot
      rot_diff[rot_diff // -self.drot > 0] = -self.drot
      rot_diff[np.abs(rot_diff) < self.drot] = 0

      if np.all(pos_diff == 0) and np.all(rot_diff == 0):
        primitive = constants.PICK_PRIMATIVE
      else:
        primitive = constants.PLACE_PRIMATIVE

      x, y, z, r = pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]

    else:
      x, y, z = 0, 0, self.dpos
      r = 0
      primitive = constants.PICK_PRIMATIVE
    return self.encodeAction(primitive, x, y, z, r)

  def getStepsLeft(self):
    return 100