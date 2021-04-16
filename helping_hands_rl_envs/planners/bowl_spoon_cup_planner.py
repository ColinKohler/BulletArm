import numpy as np
import pybullet as pb
from copy import copy
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner

class BowlSpoonCupPlanner(BlockStackingPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def getPickingAction(self):
    if not self.env.objects[0].isTouching(self.env.blanket):
      return self.pickTallestObjOnTop(objects=[self.env.objects[0]])
    elif not self.env.objects[1].isTouching(self.env.objects[0]):
      return self.pickTallestObjOnTop(objects=[self.env.objects[1]])
    else:
      return self.pickTallestObjOnTop(objects=[self.env.objects[2]])

  def getPlacingAction(self):
    if self.getHoldingObjType() is constants.BOWL:
      pos = copy(self.env.blanket_pos[:2])
      pos[0] += 0.09 * np.cos(self.env.blanket_rz)
      pos[1] += 0.09 * np.sin(self.env.blanket_rz)
      x, y, z = pos[0], pos[1], self.env.place_offset
      rx, ry, rz = 0, 0, self.env.blanket_rz
      return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, ry, rx))
    elif self.getHoldingObjType() is constants.SPOON:
      pos = self.env.objects[0].getPosition()
      x, y, z = pos[0], pos[1], pos[2]
      x += 0.02 * np.cos(self.env.blanket_rz)
      y += 0.02 * np.sin(self.env.blanket_rz)
      rx, ry, rz = 0, 0, self.env.blanket_rz + np.pi/2
      return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, ry, rx))
    else:
      pos = copy(self.env.blanket_pos[:2])
      pos[0] -= 0.06 * np.cos(self.env.blanket_rz)
      pos[1] -= 0.06 * np.sin(self.env.blanket_rz)
      x, y, z = pos[0], pos[1], self.env.place_offset
      rx, ry, rz = 0, 0, self.env.blanket_rz+np.pi/2
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
