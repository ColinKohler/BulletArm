import numpy as np

from bulletarm.pybullet.utils import constants
from bulletarm.planners.close_loop_planner import CloseLoopPlanner
from bulletarm.pybullet.utils import transformations

class CloseLoopHouseholdPickingClutteredPlanner(CloseLoopPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)
    self.current_target = None
    self.target_object = None
    self.stage = 0

  def getNextActionToCurrentTarget(self):
    x, y, z, r = self.getActionByGoalPose(self.current_target[0], self.current_target[1])
    if np.all(np.abs([x, y, z]) < self.dpos) and np.abs(r) < self.drot:
      primitive = constants.PICK_PRIMATIVE if self.current_target[2] is constants.PICK_PRIMATIVE else constants.PLACE_PRIMATIVE
      self.current_target = None
    else:
      primitive = constants.PICK_PRIMATIVE if self.stage == 0 else constants.PLACE_PRIMATIVE
    return self.env._encodeAction(primitive, x, y, z, r)

  def setNewTarget(self):
    if self.stage == 0:
      objects = np.array(list(filter(lambda x: not self.isObjectHeld(x) and self.isObjOnTop(x), self.env.objects)))
      # object_poses = self.env.getObjectPoses(objects)
      # sorted_inds = np.flip(np.argsort(object_poses[:,2], axis=0))
      # objects = objects[sorted_inds]
      np.random.shuffle(objects)
      self.target_object = objects[0]

      object_pos = list(self.target_object.getPosition())
      object_pos[2] += (np.random.random() - 1) * 0.02
      object_rot = list(transformations.euler_from_quaternion(self.target_object.getRotation()))
      rz = (np.random.random() - 0.5) * np.pi
      # while object_rot[2] < -np.pi/2:
      #   object_rot[2] += np.pi
      # while object_rot[2] > np.pi/2:
      #   object_rot[2] -= np.pi
      object_rot[2] = rz
      self.current_target = (object_pos, object_rot, constants.PICK_PRIMATIVE)
      self.stage = 1
    else:
      object_pos = self.target_object.getPosition()
      object_rot = list(transformations.euler_from_quaternion(self.target_object.getRotation()))
      object_rot[2] = transformations.euler_from_quaternion(self.env.robot._getEndEffectorRotation())[2]
      self.current_target = ((object_pos[0], object_pos[1], 0.2), object_rot, constants.PICK_PRIMATIVE)
      self.stage = 0

  def getNextAction(self):
    if self.env.current_episode_steps == 1 or self.env.grasp_done == 1:
      self.stage = 0
      self.current_target = None

    if self.current_target is not None:
      return self.getNextActionToCurrentTarget()
    else:
      self.setNewTarget()
      return self.getNextActionToCurrentTarget()

  def getStepsLeft(self):
    return 100