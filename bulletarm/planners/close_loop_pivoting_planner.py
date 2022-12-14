import copy
import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopPivotingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.stage = 0
    self.current_target = None

  def getNextActionToCurrentTarget(self):
    print(self.stage)
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    primitive = self.current_target[2]
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      self.current_target = None
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    pre_pivoting_pos, pre_pivoting_rot = self.env.pivot.getPivotingPose()
    pre_pivoting_pos[2] += 0.1
    pre_pivoting_rot = list(transformations.euler_from_quaternion(pre_pivoting_rot))

    pivoting_pos, pivoting_rot = self.env.pivot.getPivotingPose()
    pivoting_rot = list(transformations.euler_from_quaternion(pivoting_rot))

    lift_1_pos, lift_1_rot = self.env.pivot.getLift1Pose()
    lift_1_rot = list(transformations.euler_from_quaternion(lift_1_rot))

    lift_2_pos, lift_2_rot = self.env.pivot.getLift2Pose()
    lift_2_rot = list(transformations.euler_from_quaternion(lift_2_rot))

    lift_3_pos, lift_3_rot = self.env.pivot.getLift3Pose()
    lift_3_rot = list(transformations.euler_from_quaternion(lift_3_rot))

    lift_4_pos, lift_4_rot = self.env.pivot.getLift4Pose()
    lift_4_rot = list(transformations.euler_from_quaternion(lift_4_rot))

    lift_5_pos, lift_5_rot = self.env.pivot.getLift5Pose()
    lift_5_rot = list(transformations.euler_from_quaternion(lift_5_rot))

    lift_6_pos, lift_6_rot = self.env.pivot.getLift6Pose()
    lift_6_rot = list(transformations.euler_from_quaternion(lift_6_rot))

    lift_7_pos, lift_7_rot = self.env.pivot.getLift7Pose()
    lift_7_rot = list(transformations.euler_from_quaternion(lift_7_rot))

    if self.stage == 0:
      self.stage = 1
      self.current_target = (pre_pivoting_pos, pre_pivoting_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 1:
      self.stage = 2
      self.current_target = (pivoting_pos, pivoting_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 2:
      self.stage = 3
      self.current_target = (lift_1_pos, lift_1_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 3:
      self.stage = 4
      self.current_target = (lift_2_pos, lift_2_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 4:
      self.stage = 5
      self.current_target = (lift_3_pos, lift_3_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 5:
      self.stage = 6
      self.current_target = (lift_4_pos, lift_4_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 6:
      self.stage = 7
      self.current_target = (lift_5_pos, lift_5_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 7:
      self.stage = 8
      self.current_target = (lift_6_pos, lift_6_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 8:
      self.stage = 0
      self.current_target = (lift_7_pos, lift_7_rot, constants.PICK_PRIMATIVE)

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()
