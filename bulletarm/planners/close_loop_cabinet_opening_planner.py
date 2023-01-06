import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopCabinetOpeningPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.stage = 0
    self.current_target = None

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    primitive = self.current_target[2]
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      self.current_target = None
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    handle_pos = self.env.cabinet.getLeftHandlePos()
    handle_rot = transformations.euler_from_quaternion(self.env.cabinet.getLeftHandleRot())[2]
    handle_rot += np.pi / 2
    pre_pos = np.copy(handle_pos)
    pre_pos[2] += 0.12

    pull_pos = np.copy(handle_pos)
    pull_pos[0] -= 0.1

    handle_rot += np.pi/2
    while handle_rot > np.pi/2:
      handle_rot -= np.pi
    while handle_rot < -np.pi/2:
      handle_rot += np.pi
    rot = [0, 0, handle_rot]

    if self.stage == 0:
      # moving to pre
      self.stage = 1
      self.current_target = (pre_pos, rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 1:
      # moving to handle
      self.stage = 2
      self.current_target = (handle_pos, rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 2:
      # moving to handle
      self.stage = 3
      self.current_target = (handle_pos, rot, constants.PICK_PRIMATIVE)
    elif self.stage == 3:
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
