import numpy as np
import numpy.random as npr

from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants

class RampBlockStackingPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(RampBlockStackingPlanner, self).__init__(env, config)

  # def placeOnHighestObj(self, objects=None):
  #   """
  #   place on the highest object
  #   :param objects: pool of objects
  #   :return: encoded action
  #   """
  #   if objects is None: objects = self.env.objects
  #   objects, object_poses = self.getSortedObjPoses(objects=objects)
  #   x, y, z, rz = object_poses[0][0], object_poses[0][1], object_poses[0][2]+self.env.place_offset, object_poses[0][5]
  #   for obj, pose in zip(objects, object_poses):
  #     if not self.isObjectHeld(obj):
  #       x, y, z, rz = pose[0], pose[1], pose[2]+self.env.place_offset, pose[5]
  #       break
  #   rx = self.env.pick_rx
  #
  #   return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, rx))

  # def placeOnGround(self, padding_dist, min_dist):
  #   """
  #   place on the ground, avoiding all existing objects
  #   :param padding_dist: padding dist for getting valid pos
  #   :param min_dist: min dist to adjacent object
  #   :return: encoded action
  #   """
  #   existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))]
  #   try:
  #     place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1, sample_range=[self.env.workspace[0], [self.env.workspace[1][0], 0.1]])[0]
  #   except NoValidPositionException:
  #     place_pos = self.getValidPositions(padding_dist, min_dist, [], 1, sample_range=[self.env.workspace[0], [self.env.workspace[1][0], 0.1]])[0]
  #   x, y, z, rz = place_pos[0], place_pos[1], self.env.place_offset, np.pi*np.random.random_sample()
  #   rx = self.env.pick_rx
  #
  #   return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, rx))

  def getPickingAction(self):
    objects = list(filter(lambda o: not self.env.isPosOffRamp(o.getPosition()), self.env.objects))
    if len(objects) == 0:
      objects = self.env.objects

    objects, object_poses = self.getSortedObjPoses(objects=objects, roll=True)

    x, y, z, rx, ry, rz = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][3], object_poses[0][4], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, rx, ry, rz = pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, (rz, ry, rx))

  def getPlacingAction(self):
    objects = list(filter(lambda o: self.env.isPosOffRamp(o.getPosition()), self.env.objects))
    if objects:
      return self.placeOnHighestObj(objects)
    else:
      return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)

  def getStepsLeft(self):
    if not self.isSimValid():
      return 100
    return self.getNumTopBlock() - 1
