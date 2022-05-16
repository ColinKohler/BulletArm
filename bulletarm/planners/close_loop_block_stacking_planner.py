import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopBlockStackingPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.pick_place_stage = 0
    # self.post_pose_reached = False
    # pos, rot, primitive
    self.current_target = None
    self.target_obj = None

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
      self.current_target = None
    else:
      primitive = constants.PICK_PRIMATIVE if self.isHolding() else constants.PLACE_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    objects = np.array(list(filter(lambda x: not self.isObjectHeld(x) and self.isObjOnTop(x), self.env.objects)))
    object_poses = self.env.getObjectPoses(objects)
    sorted_inds = np.flip(np.argsort(object_poses[:,2], axis=0))
    objects = objects[sorted_inds]

    if self.env.current_episode_steps == 1:
      self.pick_place_stage = 0

    if self.pick_place_stage in [0, 1, 2]:
      if self.target_obj is None:
        self.target_obj = objects[1] if len(objects) > 1 else objects[0]
      object_pos = self.target_obj.getPosition()
      object_rot = list(transformations.euler_from_quaternion(self.target_obj.getRotation()))
      gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
      while object_rot[2] - gripper_rz > np.pi / 4:
        object_rot[2] -= np.pi / 2
      while object_rot[2] - gripper_rz < -np.pi / 4:
        object_rot[2] += np.pi / 2
      pre_pick_pos = object_pos[0], object_pos[1], object_pos[2] + 0.1
      if self.pick_place_stage == 0:
        self.pick_place_stage = 1
        self.current_target = (pre_pick_pos, object_rot, constants.PLACE_PRIMATIVE)
      elif self.pick_place_stage == 1:
        self.pick_place_stage = 2
        self.current_target = (object_pos, object_rot, constants.PICK_PRIMATIVE)
      else:
        self.pick_place_stage = 3
        self.target_obj = None
        self.current_target = (pre_pick_pos, object_rot, constants.PICK_PRIMATIVE)

    else:
      if self.target_obj is None:
        self.target_obj = objects[0]
      object_pos = self.target_obj.getPosition()
      object_rot = list(transformations.euler_from_quaternion(self.target_obj.getRotation()))
      gripper_rz = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
      while object_rot[2] - gripper_rz > np.pi / 4:
        object_rot[2] -= np.pi / 2
      while object_rot[2] - gripper_rz < -np.pi / 4:
        object_rot[2] += np.pi / 2
      pre_place_pos = object_pos[0], object_pos[1], object_pos[2] + 0.1
      if self.pick_place_stage == 3:
        self.pick_place_stage = 4
        self.current_target = (pre_place_pos, object_rot, constants.PICK_PRIMATIVE)
      elif self.pick_place_stage == 4:
        self.pick_place_stage = 5
        place_pos = object_pos[0], object_pos[1], object_pos[2] + self.getMaxBlockSize() * 1.2
        self.current_target = (place_pos, object_rot, constants.PLACE_PRIMATIVE)
      else:
        self.pick_place_stage = 0
        self.target_obj = None
        self.current_target = (pre_place_pos, object_rot, constants.PLACE_PRIMATIVE)

  def getNextAction(self):
    if self.env.current_episode_steps == 1:
      self.pick_place_stage = 0
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
