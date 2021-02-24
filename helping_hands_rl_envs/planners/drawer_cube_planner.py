import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class DrawerCubePlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def openDrawer(self, drawer):
    handle_pos = drawer.getHandlePosition()
    rot = (0, -np.pi / 2, 0)
    m = np.array(transformations.euler_matrix(*rot))[:3, :3]
    pos = handle_pos - m[:, 2] * 0.02

    return self.encodeAction(constants.PULL_PRIMATIVE, pos[0], pos[1], pos[2], rot)

  def pickObj(self, obj):
    obj_pos, obj_rot = obj.getPose()
    obj_rot = self.env.convertQuaternionToEuler(obj_rot)
    x, y, z, r = obj_pos[0], obj_pos[1], obj_pos[2]+self.env.pick_offset, obj_rot
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, (r[2], r[1], r[0]))


  def getNextAction(self):
    if self.env._isHolding():
      return self.encodeAction(constants.PLACE_PRIMATIVE, 0.32, 0, 0.03, 0)

    elif not self.env.drawer.isDrawerOpen():
      return self.openDrawer(self.env.drawer)

    elif self.env.drawer.isObjInsideDrawer(self.getObjects()[0]):
      return self.pickObj(self.getObjects()[0])

    elif not self.env.drawer2.isDrawerOpen():
      return self.openDrawer(self.env.drawer2)

    else:
      return self.pickObj(self.getObjects()[0])

  def getStepLeft(self):
    return 100