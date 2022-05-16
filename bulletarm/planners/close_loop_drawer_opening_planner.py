import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopDrawerOpeningPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.stage = 0
    self.current_target = None

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      self.current_target = None
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    handle_pos = self.env.drawer.getHandlePosition()
    drawer_rot = self.env.drawer_rot
    m = np.array(transformations.euler_matrix(0, 0, drawer_rot))[:3, :3]
    handle_pos = handle_pos + m[:, 0] * 0.02
    handle_pos = handle_pos - m[:, 2] * 0.02
    pull_pos = handle_pos - m[:, 0] * 0.2
    pre_pos = np.copy(handle_pos)
    pre_pos[2] += 0.1

    drawer_rot += np.pi/2
    while drawer_rot > np.pi/2:
      drawer_rot -= np.pi
    while drawer_rot < -np.pi/2:
      drawer_rot += np.pi
    rot = [0, 0, drawer_rot]

    if self.stage == 0:
      # moving to pre
      self.stage = 1
      self.current_target = (pre_pos, rot, constants.PICK_PRIMATIVE)
    elif self.stage == 1:
      # moving to handle
      self.stage = 2
      self.current_target = (handle_pos, rot, constants.PICK_PRIMATIVE)
    elif self.stage == 2:
      self.stage = 0
      self.current_target = (pull_pos, rot, constants.PICK_PRIMATIVE)

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100
