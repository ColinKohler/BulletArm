import numpy as np
import numpy.random as npr
import scipy

from helping_hands_rl_envs.envs.pybullet_env import NoValidPositionException

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants

class BlockStructurePlanner(BasePlanner):
  def __init__(self, env, env_config):
    super().__init__(env, env_config)

  def getNextAction(self):
    if self.isHolding():
      if npr.rand() < self.rand_pick_prob:
        return self.getRandomPickingAction()
      else:
        return self.getPlacingAction()
    else:
      if npr.rand() < self.rand_place_prob:
        return self.getRandomPlacingAction()
      else:
        return self.getPickingAction()

  def getRandomPickingAction(self):
    x = npr.uniform(self.env.workspace[0, 0] + 0.025, self.env.workspace[0, 1] - 0.025)
    y = npr.uniform(self.env.workspace[1, 0] + 0.025, self.env.workspace[1, 1] - 0.025)
    z = 0.
    r = npr.uniform(0., np.pi)
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getRandomPlacingAction(self):
    x = npr.uniform(self.env.workspace[0, 0], self.env.workspace[0, 1])
    y = npr.uniform(self.env.workspace[1, 0], self.env.workspace[1, 1])
    z = 0.
    r = npr.uniform(0., np.pi)
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def getPickingAction(self):
    raise NotImplemented('Planners must implement this function')

  def getPlacingAction(self):
    raise NotImplemented('Planners must implement this function')

  def pickSecondTallestObjOnTop(self, objects=None, side_grasp=False):
    """
    pick up the second highest object that is on top
    :param objects: pool of objects
    :param side_grasp: grasp on the side of the object (90 degree), should be true for triangle, brick, etc
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(roll=True, objects=objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2]+self.env.pick_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2]+self.env.pick_offset, pose[5]
        break
    if side_grasp:
      r += np.pi / 2
      while r < 0:
        r += np.pi
      while r > np.pi:
        r -= np.pi
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def placeOnHighestObj(self, objects=None, side_place=False):
    """
    place on the highest object
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)
    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2]+self.env.place_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if not self.isObjectHeld(obj):
        x, y, z, r = pose[0], pose[1], pose[2]+self.env.place_offset, pose[5]
    if side_place:
      r += np.pi / 2
      while r < 0:
        r += np.pi
      while r > np.pi:
        r -= np.pi
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeOnGround(self, padding_dist, min_dist):
    """
    place on the ground, avoiding all existing objects
    :param padding_dist: padding dist for getting valid pos
    :param min_dist: min dist to adjacent object
    :return: encoded action
    """
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))]
    place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1)[0]
    x, y, z, r = place_pos[0], place_pos[1], self.env.place_offset, 0
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeNearAnother(self, another_obj, min_dist_to_another, max_dist_to_another, padding_dist, min_dist):
    """
    place near another object, avoid existing objects except for another_obj
    :param another_obj: the object to place near to
    :param min_dist_to_another: min distance to another_obj
    :param max_dist_to_another: max distance to another_obj
    :param padding_dist: padding dist for getting valid pos
    :param min_dist: min dist to other adjacent object
    :return: encoded action
    """
    place_pos = self.getValidPositions(self.env.max_block_size * 2, self.env.max_block_size * 2, [], 1)[0]
    another_obj_position = another_obj.getPosition()
    sample_range = [[another_obj_position[0] - max_dist_to_another, another_obj_position[0] + max_dist_to_another],
                    [another_obj_position[1] - max_dist_to_another, another_obj_position[1] + max_dist_to_another]]
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x) and not another_obj == x, self.env.objects))]
    for i in range(10000):
      try:
        place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(another_obj_position[:-1]) - np.array(place_pos))
      if min_dist_to_another < dist < max_dist_to_another:
        break
    x, y, z, r = place_pos[0], place_pos[1], self.env.place_offset, 0
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress([x, another_obj_position[0]], [y, another_obj_position[1]])
    r = np.arctan(slope) + np.pi / 2
    while r > np.pi:
      r -= np.pi
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeOnTopOfMultiple(self, above_objs):
    """
    place on top of multiple objects. will calculate the slope of those objects and match the rotation
    :param above_objs: list of objects to put on top of
    :return: encoded action
    """
    obj_positions = np.array([o.getPosition() for o in above_objs])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y, z = obj_positions.mean(0)
    z += +self.env.place_offset
    r = np.arctan(slope) + np.pi / 2
    while r > np.pi:
      r -= np.pi
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def getSortedObjPoses(self, roll=False, objects=None):
    if objects is None: objects = self.env.objects
    objects = np.array(list(filter(lambda x: not self.isObjectHeld(x), objects)))
    object_poses = self.env.getObjectPoses(objects)

    # Sort by block size
    sorted_inds = np.flip(np.argsort(object_poses[:,2], axis=0))

    # TODO: Should get a better var name for this
    if roll:
      sorted_inds = np.roll(sorted_inds, -1)

    objects = objects[sorted_inds]
    object_poses = object_poses[sorted_inds]
    return objects, object_poses
