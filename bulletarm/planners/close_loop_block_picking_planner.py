import copy
import numpy as np
import numpy.random as npr

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopBlockPickingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

    self.stage = 0
    self.current_trarget = None

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
      self.current_target = None
    else:
      primitive = constants.PICK_PRIMATIVE if self.isHolding() else constants.PLACE_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    block_pos = self.env.objects[0].getPosition()
    block_rot = list(transformations.euler_from_quaternion(self.env.objects[0].getRotation()))

    pre_grasp_pos = copy.copy(block_pos)
    pre_grasp_pos[0] += npr.uniform(-0.02, 0.02)
    pre_grasp_pos[1] += npr.uniform(-0.02, 0.02)
    pre_grasp_pos[2] += npr.uniform(0.08, 0.16)
    pre_grasp_rot = block_rot

    grasp_pos = copy.copy(block_pos)
    grasp_pos[0] += npr.uniform(-0.01, 0.01)
    grasp_pos[1] += npr.uniform(-0.01, 0.01)
    grasp_pos[2] += npr.uniform(-0.02, 0.01)
    grasp_rot = block_rot

    post_grasp_pos = copy.copy(block_pos)
    post_grasp_pos[0] += npr.uniform(-0.02, 0.02)
    post_grasp_pos[1] += npr.uniform(-0.02, 0.02)
    post_grasp_pos[2] += 0.12

    post_grasp_rot = block_rot

    if self.stage == 0:
      self.stage = 1
      self.current_target = (pre_grasp_pos, pre_grasp_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 1:
      self.stage = 2
      self.current_target = (grasp_pos, grasp_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 2:
      self.stage = 0
      self.current_target = (post_grasp_pos, post_grasp_rot, constants.PICK_PRIMATIVE)

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

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

class CloseLoopBlockPickingPlanner2(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

    self.stage = 0
    self.current_trarget = None

    self.i = -1
    self.gf = [[0,0], [0, 1], [1,0], [1,1]]

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
      self.current_target = None
    else:
      primitive = constants.PICK_PRIMATIVE if self.isHolding() else constants.PLACE_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    if self.stage == 0:
      self.grasp_orientation, self.finger_to_collide = self.gf[self.i % 4]
      #self.grasp_orientation = 0 if npr.rand() < 0.5 else 1
      #self.finger_to_collide = 0 if npr.rand() < 0.5 else 1

    block_pos = self.env.objects[0].getPosition()
    block_rot = list(transformations.euler_from_quaternion(self.env.objects[0].getRotation()))
    block_rot[2] = block_rot[2] if self.grasp_orientation else block_rot[2] + np.radians(90)

    pre_grasp_pos = copy.copy(block_pos)
    pre_grasp_pos[2] += 0.12
    pre_grasp_rot = block_rot

    grasp_pos = copy.copy(block_pos)
    grasp_pos[0] += npr.uniform(-0.03, 0.03)
    grasp_pos[1] += npr.uniform(-0.03, 0.03)
    grasp_pos[2] -= 0.02
    grasp_rot = block_rot

    post_grasp_pos = copy.copy(block_pos)
    post_grasp_pos[2] += 0.12
    post_grasp_rot = block_rot

    if self.stage == 0:
      self.stage = 1
      self.current_target = (pre_grasp_pos, pre_grasp_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 1:
      self.stage = 2
      self.current_target = (grasp_pos, grasp_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 2:
      self.stage = 0
      self.current_target = (post_grasp_pos, post_grasp_rot, constants.PICK_PRIMATIVE)

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.i += 1
      self.stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

  def getActionByGoalPose(self, goal_pos, goal_rot):
    current_pos = self.env.robot._getFingerPositions()[self.finger_to_collide]
    current_rot = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())
    pos_diff = goal_pos - current_pos
    rot_diff = np.array(goal_rot) - current_rot

    pos_diff[pos_diff // self.dpos > 0] = self.dpos
    pos_diff[pos_diff // -self.dpos > 0] = -self.dpos

    rot_diff[rot_diff // self.drot > 0] = self.drot
    rot_diff[rot_diff // -self.drot > 0] = -self.drot

    x, y, z, r = pos_diff[0], pos_diff[1], pos_diff[2], rot_diff[2]
    return x, y, z, r
