import copy
import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.planners.close_loop_planner import CloseLoopPlanner
from helping_hands_rl_envs.pybullet.utils import transformations

class CloseLoopPegInsertionPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.stage = 0
    self.current_target = None

  def getNextActionToCurrentTarget(self):
    ee_to_peg = self.env.peg.getPosition() - self.env.robot._getEndEffectorPosition()
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    #x += ee_to_peg[0]
    #y += ee_to_peg[1]

    primitive = constants.PICK_PRIMATIVE
    pos_min = self.current_target[3] if self.current_target[3] is not None else self.dpos
    rot_min = self.current_target[4] if self.current_target[4] is not None else self.dpos
    if np.all(np.abs([x, y, z]) < pos_min) and np.abs(r) < rot_min:
      self.current_target = None

    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    peg_pos, peg_rot = self.env.peg_hole.getHolePose()

    #hole_z_offset = 0.082
    hole_z_offset = 0.078

    drag_pos_1 = copy.copy(peg_pos)
    drag_pos_1[2] += 0.15
    drag_rot_1 = list(transformations.euler_from_quaternion(peg_rot))

    drag_pos_2 = copy.copy(peg_pos)
    drag_pos_2[2] += hole_z_offset
    drag_rot_2 = list(transformations.euler_from_quaternion(peg_rot))

    pre_insert_pos, pre_insert_rot = peg_pos, peg_rot
    pre_insert_pos[2] += hole_z_offset
    pre_insert_rot = list(transformations.euler_from_quaternion(pre_insert_rot))

    insert_pos, insert_rot = self.env.peg_hole.getHolePose()
    insert_rot = list(transformations.euler_from_quaternion(insert_rot))

    gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]

    if self.stage == 0:
      self.hole_x_offset = npr.uniform(-0.05, 0.05)
      self.hole_y_offset = npr.uniform(-0.05, 0.05)
      drag_pos_1[0] += self.hole_x_offset
      drag_pos_1[1] += self.hole_y_offset

      # Place peg on insert box
      while drag_rot_1[2] - gripper_rz > np.pi/4:
        drag_rot_1[2] -= np.pi/2
      while drag_rot_1[2] - gripper_rz < -np.pi/4:
        drag_rot_1[2] += np.pi/2

      self.stage = 1
      self.current_target = (drag_pos_1, drag_rot_1, constants.PICK_PRIMATIVE, None, None)
    elif self.stage == 1:
      # Place peg on insert box
      while drag_rot_2[2] - gripper_rz > np.pi/4:
        drag_rot_2[2] -= np.pi/2
      while drag_rot_2[2] - gripper_rz < -np.pi/4:
        drag_rot_2[2] += np.pi/2

      drag_pos_2[0] += self.hole_x_offset
      drag_pos_2[1] += self.hole_y_offset

      self.stage = 2
      self.current_target = (drag_pos_2, drag_rot_2, constants.PICK_PRIMATIVE, 5e-3, None)
    elif self.stage == 2:
      # Drag peg to hole
      while pre_insert_rot[2] - gripper_rz > np.pi/4:
        pre_insert_rot[2] -= np.pi/2
      while pre_insert_rot[2] - gripper_rz < -np.pi/4:
        pre_insert_rot[2] += np.pi/2

      self.stage = 3
      self.current_target = (pre_insert_pos, pre_insert_rot, constants.PICK_PRIMATIVE, 5e-3, 1e-1)
    elif self.stage == 3:
      # insert peg
      while insert_rot[2] - gripper_rz > np.pi/4:
        insert_rot[2] -= np.pi/2
      while insert_rot[2] - gripper_rz < -np.pi/4:
        insert_rot[2] += np.pi/2

      self.stage = 0
      self.current_target = (insert_pos, insert_rot, constants.PICK_PRIMATIVE, 5e-3, 1e-1)

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
