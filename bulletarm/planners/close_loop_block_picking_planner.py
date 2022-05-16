import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopBlockPickingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def getNextAction(self):
    if not self.env._isHolding():
      block_pos = self.env.objects[0].getPosition()
      block_rot = transformations.euler_from_quaternion(self.env.objects[0].getRotation())

      x, y, z, r = self.getActionByGoalPose(block_pos, block_rot)

      if np.all(np.abs([x, y, z]) < self.dpos) and (not self.random_orientation or np.abs(r) < self.drot):
        primitive = constants.PICK_PRIMATIVE
      else:
        primitive = constants.PLACE_PRIMATIVE

    else:
      x, y, z = 0, 0, self.dpos
      r = 0
      primitive = constants.PICK_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def getStepsLeft(self):
    return 100
