import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopBlockPickingCornerPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.stage = 0 # 1: approaching pre, 2: pre->press 3: pull 4: pick
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
    object_pos = self.env.objects[0].getPosition()
    object_rot = list(transformations.euler_from_quaternion(self.env.objects[0].getRotation()))
    while object_rot[2] > np.pi/2:
      object_rot[2] -= np.pi
    while object_rot[2] < -np.pi/2:
      object_rot[2] += np.pi

    pull_rz = self.env.corner_rz + np.pi/4
    while pull_rz > np.pi/2:
      pull_rz -= np.pi
    while pull_rz < -np.pi/2:
      pull_rz += np.pi

    pre_press_pos = self.env.corner.getPressPose()[0]
    pre_press_pos[2] = 0.1

    pre_press_rot = [0, 0, pull_rz]

    press_pos = self.env.corner.getPressPose()[0]
    press_pos[2] += 0.4 * self.env.max_block_size

    pull_pos = self.env.corner.getPullPose()[0]
    pull_pos[2] += 0.4 * self.env.max_block_size

    pull_rot = [0, 0, pull_rz]

    post_pull_pos = self.env.corner.getPullPose()[0]
    post_pull_pos[2] = 0.1

    pre_pick_pos = object_pos[0], object_pos[1], 0.1
    if self.stage == 0:
      # moving to pre press
      self.stage = 1
      self.current_target = (pre_press_pos, pre_press_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 1:
      # moving to press
      self.stage = 2
      self.current_target = (press_pos, pre_press_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 2:
      # moving to pull
      self.stage = 3
      self.current_target = (pull_pos, pull_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 3:
      # moving to pre pick
      self.stage = 4
      self.current_target = (post_pull_pos, pull_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 4:
      self.stage = 5
      self.current_target = (pre_pick_pos, object_rot, constants.PLACE_PRIMATIVE)
    elif self.stage == 5:
      # moving to pick
      self.stage = 6
      self.current_target = (object_pos, object_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 6:
      self.stage = 0
      self.current_target = (pre_pick_pos, object_rot, constants.PICK_PRIMATIVE)

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.stage = 0
      self.current_target = None
    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()


    # if not self.env._isHolding():
    #   self.pre_pose_reached = False
    #   block_pos = self.env.objects[0].getPosition()
    #   block_rot = transformations.euler_from_quaternion(self.env.objects[0].getRotation())
    #
    #   x, y, z, r = self.getActionByGoalPose(block_pos, block_rot)
    #
    #   if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
    #     primitive = constants.PICK_PRIMATIVE
    #   else:
    #     primitive = constants.PLACE_PRIMATIVE
    #
    # else:
    #   block_pos = self.env.objects[1].getPosition()
    #   block_rot = transformations.euler_from_quaternion(self.env.objects[1].getRotation())
    #
    #   pre_place_pos = block_pos[0], block_pos[1], 0.1
    #   x, y, z, r = self.getActionByGoalPose(pre_place_pos, block_rot)
    #   primitive = constants.PICK_PRIMATIVE
    #   if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
    #     self.pre_pose_reached = True
    #
    #   if self.pre_pose_reached:
    #     place_pos = block_pos[0], block_pos[1], block_pos[2] + self.getMaxBlockSize()
    #     x, y, z, r = self.getActionByGoalPose(place_pos, block_rot)
    #     if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi / 12:
    #       primitive = constants.PLACE_PRIMATIVE
    #
    #
    #   # if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
    #   #   self.pre_pose_reached = True
    #   #   place_pos = block_pos[0], block_pos[1], block_pos[2] + self.getMaxBlockSize()/2
    #   #   x, y, z, r = self.getActionByGoalPose(place_pos, block_rot)
    #   #   if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
    #   #     primitive = constants.PLACE_PRIMATIVE
    # return self.env._encodeAction(primitive, x, y, z, r)

  def getStepsLeft(self):
    return 100
