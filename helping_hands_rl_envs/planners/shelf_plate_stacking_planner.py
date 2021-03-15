import numpy as np
import pybullet as pb
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner

class ShelfPlateStackingPlanner(BasePlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def getNextAction(self):
    if self.isHolding():
      return self.getPlacingAction()
    else:
      return self.getPickingAction()


  def getPickingAction(self):
    objects = self.env.objects

    objects, object_poses = self.getSortedObjPoses(objects=objects, roll=True)

    x, y, z, rx, ry, rz = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][3], object_poses[0][4], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, rx, ry, rz = pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, (rz, ry, rx))

  def getPlacingAction(self):
    if not self.env.anyObjectOnTarget1():
      return self.placeOnShelf()

    objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)
    x, y, z, rx, ry, rz = object_poses[0][0], object_poses[0][1], object_poses[0][2] + self.env.place_offset, 0, self.env.place_ry_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if not self.isObjectHeld(obj):
        x, y, z, rx, ry, rz = pose[0], pose[1], pose[2] + self.env.place_offset, 0, self.env.place_ry_offset, pose[5]
        break
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, ry, rx))

  def placeOnShelf(self):
    x, y, z = self.env.shelf.getTarget1Pos()
    x -= 0.09
    z += self.env.place_offset
    rz, ry, rx = np.pi, self.env.place_ry_offset, 0
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, ry, rx))

  def getSortedObjPoses(self, roll=False, objects=None, ascend=False):
    if objects is None: objects = self.env.objects
    objects = np.array(list(filter(lambda x: not self.isObjectHeld(x), objects)))
    object_poses = list()
    for obj in objects:
      if self.isObjectHeld(obj):
        continue
      pos, rot = obj.getGraspPose()
      rot = self.env.convertQuaternionToEuler(rot)

      object_poses.append(pos + rot)
    object_poses = np.array(object_poses)

    # Sort by block size
    if ascend:
      sorted_inds = np.argsort(object_poses[:, 2], axis=0)
    else:
      sorted_inds = np.flip(np.argsort(object_poses[:,2], axis=0))

    # TODO: Should get a better var name for this
    if roll:
      sorted_inds = np.roll(sorted_inds, -1)

    objects = objects[sorted_inds]
    object_poses = object_poses[sorted_inds]
    return objects, object_poses
