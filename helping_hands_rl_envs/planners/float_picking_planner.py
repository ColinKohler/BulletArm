import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants

class FloatPickingPlanner(BasePlanner):
  def __init__(self, env, config):
    super(FloatPickingPlanner, self).__init__(env, config)

  def getNextAction(self):
    block_poses = list(filter(lambda o: self.env._isPointInWorkspace(o), self.env.getObjectPoses()))
    if len(block_poses) == 0:
      block_poses = self.env.getObjectPoses()
    x = block_poses[0][0]
    y = block_poses[0][1]
    z = block_poses[0][2]
    rx = -block_poses[0][3]
    rz = block_poses[0][5]

    if self.pos_noise: x, y = self.addNoiseToPos(x, y)
    if self.rot_noise:
      rz = self.addNoiseToRot(rz)
      rx = self.addNoiseToRot(rx)

    return self.env._encodeAction(constants.PICK_PRIMATIVE, x, y, z, (rz, rx))

  def getStepsLeft(self):
    if not self.env.isSimValid():
      return 100
    step_left = self.env.num_obj - self.env.obj_grasped
    return step_left