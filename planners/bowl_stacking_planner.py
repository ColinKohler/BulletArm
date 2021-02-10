import numpy as np
import pybullet as pb
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_stacking_planner import BlockStackingPlanner

class BowlStackingPlanner(BlockStackingPlanner):
  def __init__(self, env, config):
    super().__init__(env, config)

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
