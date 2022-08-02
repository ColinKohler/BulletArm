import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.planners.close_loop_planner import CloseLoopPlanner
from helping_hands_rl_envs.pybullet.utils import transformations

class CloseLoopPegInsertionPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.rand_point = config['rand_point'] if 'rand_point' in config else False
    self.stage = 0
    self.current_target = None

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    primitive = constants.PICK_PRIMATIVE
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      self.current_target = None

    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    pre_insert_pos, pre_insert_rot = self.env.peg_hole.getHolePose()
    pre_insert_pos[2] += 0.14
    pre_insert_rot = list(transformations.euler_from_quaternion(pre_insert_rot))

    insert_pos, insert_rot = self.env.peg_hole.getHolePose()
    insert_rot = list(transformations.euler_from_quaternion(insert_rot))

    gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]

    if self.stage == 0:
      self.dpos = 0.025
      # moving to pre insert
      while pre_insert_rot[2] - gripper_rz > np.pi/4:
        pre_insert_rot[2] -= np.pi/2
      while pre_insert_rot[2] - gripper_rz < -np.pi/4:
        pre_insert_rot[2] += np.pi/2

      self.stage = 1
      self.current_target = (pre_insert_pos, pre_insert_rot, constants.PICK_PRIMATIVE)
    elif self.stage == 1:
      # insert peg
      self.dpos = 0.025
      while insert_rot[2] - gripper_rz > np.pi/4:
        insert_rot[2] -= np.pi/2
      while insert_rot[2] - gripper_rz < -np.pi/4:
        insert_rot[2] += np.pi/2

      self.stage = 0
      self.current_target = (insert_pos, insert_rot, constants.PICK_PRIMATIVE)

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
