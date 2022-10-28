import copy
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopMugPickingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.stage = 0 # 1: approaching pre, 2: pick 3: lift
    self.current_target = None

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
      self.current_target = None
    else:
      primitive = constants.PICK_PRIMATIVE if self.isHolding() else constants.PLACE_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    mug_grasp_pos = self.env.objects[0].getGraspPosition()
    mug_grasp_pos[2] -= 0.02
    mug_grasp_rot = transformations.euler_from_quaternion(self.env.objects[0].getGraspRotation())

    pre_grasp_pos = copy.copy(mug_grasp_pos)
    pre_grasp_pos[2] = 0.12

    post_grasp_pos = copy.copy(mug_grasp_pos)
    post_grasp_pos[2] = 0.20

    if self.stage == 0:
      self.stage = 1
      self.current_target = (pre_grasp_pos, mug_grasp_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 1:
      self.stage = 2
      self.current_target = (mug_grasp_pos, mug_grasp_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 2:
      self.stage = 0
      self.current_target = (post_grasp_pos, mug_grasp_rot, constants.PICK_PRIMATIVE)

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()
