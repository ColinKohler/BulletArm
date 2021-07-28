import numpy as np

from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.planners.close_loop_planner import CloseLoopPlanner
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

class CloseLoopBlockStackingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.pre_pose_reached = False

  def getNextAction(self):
    if not self.env._isHolding():
      self.pre_pose_reached = False
      block_pos = self.env.objects[0].getPosition()
      block_rot = transformations.euler_from_quaternion(self.env.objects[0].getRotation())

      x, y, z, r = self.getActionByGoalPose(block_pos, block_rot)

      if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
        primitive = constants.PICK_PRIMATIVE
      else:
        primitive = constants.PLACE_PRIMATIVE

    else:
      block_pos = self.env.objects[1].getPosition()
      block_rot = transformations.euler_from_quaternion(self.env.objects[1].getRotation())

      pre_place_pos = block_pos[0], block_pos[1], 0.1
      x, y, z, r = self.getActionByGoalPose(pre_place_pos, block_rot)
      primitive = constants.PICK_PRIMATIVE
      if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
        self.pre_pose_reached = True

      if self.pre_pose_reached:
        place_pos = block_pos[0], block_pos[1], block_pos[2] + self.getMaxBlockSize()
        x, y, z, r = self.getActionByGoalPose(place_pos, block_rot)
        if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi / 12:
          primitive = constants.PLACE_PRIMATIVE


      # if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
      #   self.pre_pose_reached = True
      #   place_pos = block_pos[0], block_pos[1], block_pos[2] + self.getMaxBlockSize()/2
      #   x, y, z, r = self.getActionByGoalPose(place_pos, block_rot)
      #   if np.all(np.abs([x, y, z]) < 0.005) and np.abs(r) < np.pi/12:
      #     primitive = constants.PLACE_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def getStepsLeft(self):
    return 100