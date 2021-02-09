import numpy as np
import numpy.random as npr
import scipy

from helping_hands_rl_envs.planners.base_planner import BasePlanner
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException

import pybullet as pb

class BlockStructureBasePlanner(BasePlanner):
  def __init__(self, env, env_config):
    super().__init__(env, env_config)

  def getNextAction(self):
    if self.isHolding():
      if npr.rand() < self.rand_place_prob:
        return self.getRandomPlacingAction()
      else:
        return self.getPlacingAction()
    else:
      if npr.rand() < self.rand_pick_prob:
        return self.getRandomPickingAction()
      else:
        return self.getPickingAction()

  def getPickingAction(self):
    raise NotImplemented('Planners must implement this function')

  def getPlacingAction(self):
    raise NotImplemented('Planners must implement this function')

  def pickTallestObjOnTop(self, objects=None):
    """
    pick up the highest object that is on top
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2]+self.env.pick_offset, pose[5]
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def pickSecondTallestObjOnTop(self, objects=None):
    """
    pick up the second highest object that is on top
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(roll=True, objects=objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2]+self.env.pick_offset, pose[5]
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def pickShortestObjOnTop(self, objects=None,):
    """
    pick up the shortest object that is on top
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects, ascend=True)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2], pose[5]
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def pickRandomObjOnTop(self, objects=None):
    if objects is None: objects = self.env.objects
    npr.shuffle(objects)
    object_poses = self.env.getObjectPoses(objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2], object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2], pose[5]
        break

    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def placeOnHighestObj(self, objects=None):
    """
    place on the highest object
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)
    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2]+self.env.max_block_size/2+self.env.place_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if not self.isObjectHeld(obj):
        x, y, z, r = pose[0], pose[1], pose[2]+self.env.max_block_size/2+self.env.place_offset, pose[5]
        break
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeOnGround(self, padding_dist, min_dist):
    """
    place on the ground, avoiding all existing objects
    :param padding_dist: padding dist for getting valid pos
    :param min_dist: min dist to adjacent object
    :return: encoded action
    """
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))]
    try:
      place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1)[0]
    except NoValidPositionException:
      place_pos = self.getValidPositions(padding_dist, min_dist, [], 1)[0]
    x, y, z = place_pos[0], place_pos[1], self.env.place_offset
    r = np.pi*np.random.random_sample() if self.random_orientation else 0
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeAdjacent(self, obj, min_dist, max_dist, border_padding, obj_padding):
    place_pos = self.getValidPositions(self.env.max_block_size * 2, self.env.max_block_size * 2, list(), 1)[0]
    obj_position = obj.getPosition()
    sample_range = [[obj_position[0] - max_dist, obj_position[0] + max_dist],
                    [obj_position[1] - max_dist, obj_position[1] + max_dist]]
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x) and not obj == x, self.env.objects))]

    for i in range(100):
      try:
        place_pos = self.getValidPositions(border_padding, obj_padding, existing_pos, 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(obj_position[:-1]) - np.array(place_pos))
      # if min_dist < dist < max_dist: print('{:.3f}:{:.3f} | {:.3f}:{:.3f}'.format(place_pos[0], obj_position[0], place_pos[1], obj_position[1]))
      if min_dist < dist < max_dist and (np.allclose(place_pos[0], obj_position[0], atol=0.01) or np.allclose(place_pos[1], obj_position[1], atol=0.01)):
        break

    x, y, z, r = place_pos[0], place_pos[1], self.env.place_offset, 0
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress([x, obj_position[0]], [y, obj_position[1]])
    # r = np.arctan(slope) + np.pi / 2
    # while r > np.pi:
    #   r -= np.pi

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
    if self.random_orientation:
      sample_range = [[another_obj_position[0] - max_dist_to_another, another_obj_position[0] + max_dist_to_another],
                      [another_obj_position[1] - max_dist_to_another, another_obj_position[1] + max_dist_to_another]]
    else:
      sample_range = [[another_obj_position[0] - 0.001, another_obj_position[0] + 0.001],
                      [another_obj_position[1] - max_dist_to_another, another_obj_position[1] + max_dist_to_another]]
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x) and not another_obj == x, self.env.objects))]
    for i in range(100):
      try:
        place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1, sample_range=sample_range)[0]
      except NoValidPositionException:
        continue
      dist = np.linalg.norm(np.array(another_obj_position[:-1]) - np.array(place_pos))
      if min_dist_to_another < dist < max_dist_to_another:
        break
    x, y, z, r = place_pos[0], place_pos[1], self.env.place_offset, 0
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress([x, another_obj_position[0]], [y, another_obj_position[1]])
    r = np.arctan(slope) if self.random_orientation else 0

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeOnTopOfMultiple(self, bottom_objs):
    """
    place on top of multiple objects. will calculate the slope of those objects and match the rotation
    :param bottom_objs: list of objects to put on top of
    :return: encoded action
    """
    obj_positions = np.array([o.getPosition() for o in bottom_objs])
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(obj_positions[:, 0], obj_positions[:, 1])
    x, y, z = obj_positions.mean(0)
    z += self.env.max_block_size/2+self.env.place_offset
    r = np.arctan(slope) + np.pi / 2

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def placeOn(self, bottom_obj, bottom_obj_length, num_slots=2):
    """
    place object on top of bottom_obj. will create n=num_slots placing slots on top of bottom_obj, and place current obj
    that is furthest to another object on top of bottom_obj

    this function is for creating structures s.t. several small objects will be place on top of a long, big obj
    :param bottom_obj: the object at the bottom
    :param bottom_obj_length: the length of bottom_obj
    :param num_slots: number of slots for placement
    :return: encoded action
    """
    assert num_slots > 0
    if num_slots == 1:
      return self.placeOnHighestObj([bottom_obj])
    bottom_pos, bottom_rot = bottom_obj.getPose()
    bottom_rot = pb.getEulerFromQuaternion(bottom_rot)[2]
    v = np.array([[np.cos(bottom_rot+np.pi/2), np.sin(bottom_rot+np.pi/2)]]).repeat(num_slots, 0)

    bottom_obj_length -= self.getMaxBlockSize()
    possible_points = v * np.expand_dims(np.linspace(-0.5, 0.5, num_slots), 1).repeat(2, 1) * bottom_obj_length
    possible_points[:, 0] += bottom_pos[0]
    possible_points[:, 1] += bottom_pos[1]

    top_objs = self.getObjectsOnTopOf(bottom_obj)
    if len(top_objs) == 0:
      x, y = possible_points[np.random.choice(num_slots)]

    else:
      top_obj_positions = [o.getXYPosition() for o in top_objs]
      x, y = possible_points[np.argmax(scipy.spatial.distance.cdist(possible_points, top_obj_positions).min(axis=1))]

    z = bottom_pos[2] + self.env.max_block_size/2+self.env.place_offset
    r = bottom_rot

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def getSortedObjPoses(self, roll=False, objects=None, ascend=False):
    if objects is None: objects = self.env.objects
    objects = np.array(list(filter(lambda x: not self.isObjectHeld(x), objects)))
    object_poses = self.env.getObjectPoses(objects)

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

  def isNear(self, obj_1, obj_2):
    return np.allclose(obj_1.getZPosition(), obj_2.getZPosition(), atol=0.01) and \
           self.getDistance(obj_1, obj_2) < 2.0 * self.getMaxBlockSize()

  def isAdjacent(self, objects):
    pos = np.array([o.getPosition() for o in objects])
    if (pos[:,2].max() - pos[:,2].min()) > 0.01: return False

    if np.allclose(pos[:,0], pos[0,0], atol=0.01):
      return np.abs(pos[:,1].max() - pos[:,1].min()) < self.getMaxBlockSize() * 2
    elif np.allclose(pos[:,1], pos[0,1], atol=0.01):
      return np.abs(pos[:,0].max() - pos[:,0].min()) < self.getMaxBlockSize() * 2
    else:
      return False
