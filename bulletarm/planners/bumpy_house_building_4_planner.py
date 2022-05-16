import scipy

from bulletarm.planners.house_building_4_planner import HouseBuilding4Planner
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

from itertools import permutations

import numpy.random as npr
import numpy as np

class BumpyHouseBuilding4Planner(HouseBuilding4Planner):
  def __init__(self, env, config):
    super().__init__(env, config)

  def isObjOnGround(self, obj):
    return self.env.isObjOnPlatform(obj)

  def pickSecondTallestObjOnTop(self, objects=None):
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(roll=True, objects=objects)

    x, y, z, rx, ry, rz = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][3], object_poses[0][4], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, rx, ry, rz = pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, (rz, ry, rx))

  def isPosOnPlatform(self, pos):
    return np.linalg.norm(np.array(self.env.platform_pos) - np.array(pos)) < self.env.platform_size / 2 - 0.01

  def placeNearAnotherOnPlatform(self, another_obj, min_dist_to_another, max_dist_to_another, padding_dist, min_dist):
    place_pos = self.getValidPositions(self.env.max_block_size * 2, self.env.max_block_size * 2, [], 1)[0]
    another_obj_position = another_obj.getPosition()
    if self.random_orientation:
      sample_range = [[another_obj_position[0] - max_dist_to_another, another_obj_position[0] + max_dist_to_another],
                      [another_obj_position[1] - max_dist_to_another, another_obj_position[1] + max_dist_to_another]]
    else:
      sample_range = [[another_obj_position[0] - 0.001, another_obj_position[0] + 0.001],
                      [another_obj_position[1] - max_dist_to_another, another_obj_position[1] + max_dist_to_another]]
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x) and not another_obj == x, self.env.objects))]
    for i in range(100):
      try:
        place_pos = self.getValidPositions(padding_dist, min_dist, [], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(another_obj_position[:-1]) - np.array(place_pos))
      if min_dist_to_another < dist < max_dist_to_another and self.isPosOnPlatform(place_pos):
        break
    x, y, z, r = place_pos[0], place_pos[1], self.env.place_offset + self.env.bump_offset, 0
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress([x, another_obj_position[0]], [y, another_obj_position[1]])
    r = np.arctan(slope)+np.pi/2 if self.random_orientation else 0

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeOnPlatform(self):
    sample_range = [[self.env.platform_pos[0] - self.env.platform_size, self.env.platform_pos[0] + self.env.platform_size],
                    [self.env.platform_pos[1] - self.env.platform_size, self.env.platform_pos[1] + self.env.platform_size]]
    place_pos = self.getValidPositions(0, 0, [], 1, sample_range=sample_range)[0]
    for i in range(100):
      try:
        place_pos = self.getValidPositions(0, 0, [], 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      if self.isPosOnPlatform(place_pos):
        break
    x, y, z = place_pos[0], place_pos[1], self.env.place_offset + self.env.bump_offset
    r = np.random.random() * np.pi if self.random_orientation else 0
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def getPlacingAction(self):
    level1_blocks, level2_blocks, bricks, roofs = self.getObjs()
    if not self.checkFirstLayer():
      obj_on_pt = list(filter(self.isObjOnGround, self.env.objects))
      if len(obj_on_pt) == 0:
        return self.placeOnPlatform()
      elif self.getHoldingObjType() is constants.CUBE:
        other_object = obj_on_pt[0]
        return self.placeNearAnotherOnPlatform(other_object, self.getMaxBlockSize()*1.7, self.getMaxBlockSize()*1.8, self.getMaxBlockSize()*2, self.getMaxBlockSize()*3)
      else:
        return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)
    elif not self.checkSecondLayer():
      if self.getHoldingObjType() is constants.BRICK:
        return self.placeOnTopOfMultiple(level1_blocks)
      else:
        return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)
    elif not self.checkThirdLayer():
      if self.getHoldingObjType() is constants.CUBE:
        return self.placeOn(bricks[0], 2.8 * self.getMaxBlockSize(), 2)
      else:
        return self.placeOnGround(self.getMaxBlockSize()*3, self.getMaxBlockSize()*3)
    else:
      return self.placeOnTopOfMultiple(level2_blocks)

  def checkFirstLayer(self):
    level1_blocks, level2_blocks, bricks, roofs = self.getObjs()
    if len(level1_blocks) < 2:
      return False
    block1_pos = level1_blocks[0].getPosition()
    block2_pos = level1_blocks[1].getPosition()
    return self.getDistance(level1_blocks[0], level1_blocks[1]) < 2.3 * self.getMaxBlockSize()
