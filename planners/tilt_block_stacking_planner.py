import numpy as np
import numpy.random as npr

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException
from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.planners.block_structure_base_planner import BlockStructureBasePlanner
from helping_hands_rl_envs.simulators import constants

class TiltBlockStackingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(TiltBlockStackingPlanner, self).__init__(env, config)

  def placeOnHighestObj(self, objects=None, side_place=False):
    """
    place on the highest object
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)
    x, y, z, rz = object_poses[0][0], object_poses[0][1], object_poses[0][2]+self.env.place_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if not self.isObjectHeld(obj):
        x, y, z, rz = pose[0], pose[1], pose[2]+self.env.place_offset, pose[5]
        break
    if side_place:
      rz += np.pi / 2
    rx = self.env.pick_rx

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, rx))

  def placeOnGround(self, padding_dist, min_dist):
    """
    place on the ground, avoiding all existing objects
    :param padding_dist: padding dist for getting valid pos
    :param min_dist: min dist to adjacent object
    :return: encoded action
    """
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))]
    try:
      place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1, sample_range=[self.env.workspace[0], [self.env.workspace[1][0], 0.1]])[0]
    except NoValidPositionException:
      place_pos = self.getValidPositions(padding_dist, min_dist, [], 1, sample_range=[self.env.workspace[0], [self.env.workspace[1][0], 0.1]])[0]
    x, y, z, rz = place_pos[0], place_pos[1], self.env.place_offset, np.pi*np.random.random_sample()
    rx = self.env.pick_rx

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, rx))

  def getPickingAction(self):
    objects = list(filter(lambda o: o.getPosition()[1] > -0.1, self.env.objects))
    objects, object_poses = self.getSortedObjPoses(objects=objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2] + self.env.pick_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2] + self.env.pick_offset, pose[5]
        break


    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getPlacingAction(self):
    objects = list(filter(lambda o: o.getPosition()[1] < -0.1, self.env.objects))
    if objects:
      return self.placeOnHighestObj(objects)
    else:
      return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)

  def getStepLeft(self):
    if not self.isSimValid():
      return 100
    return self.getNumTopBlock() - 1
