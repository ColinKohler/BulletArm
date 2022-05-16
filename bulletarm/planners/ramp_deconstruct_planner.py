import numpy as np
import numpy.random as npr
import pybullet as pb
from itertools import combinations

from bulletarm.planners.block_stacking_planner import BlockStackingPlanner
from bulletarm.planners.base_planner import BasePlanner
from bulletarm.planners.block_structure_base_planner import BlockStructureBasePlanner
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException
from bulletarm.pybullet.utils import pybullet_util
from bulletarm.pybullet.utils import transformations

class RampDeconstructPlanner(BlockStructureBasePlanner):
  def __init__(self, env, config):
    super(RampDeconstructPlanner, self).__init__(env, config)
    self.objs_to_remove = []

  def getStepsLeft(self):
    return 100

  def pickTallestObjOnTop(self, objects=None):
    """
    pick up the highest object that is on top
    :param objects: pool of objects
    :return: encoded action
    """
    if objects is None: objects = self.env.objects
    objects, object_poses = self.getSortedObjPoses(objects=objects)

    x, y, z, r = object_poses[0][0], object_poses[0][1], object_poses[0][2]+self.env.pick_offset, object_poses[0][5]
    for obj, pose in zip(objects, object_poses):
      if self.isObjOnTop(obj):
        x, y, z, r = pose[0], pose[1], pose[2]+self.env.pick_offset, pose[5]
        if obj in self.objs_to_remove:
          self.objs_to_remove.remove(obj)
        break
      while r < 0:
        r += 2*np.pi
      while r > 2*np.pi:
        r -= 2*np.pi
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def placeOnGround(self, padding_dist, min_dist):
    """
    place on the ground, avoiding all existing objects
    :param padding_dist: padding dist for getting valid pos
    :param min_dist: min dist to adjacent object
    :return: encoded action
    """
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))]
    for i in range(100):
      try:
        place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1)[0]
      except NoValidPositionException:
        place_pos = self.getValidPositions(padding_dist, min_dist, [], 1)[0]
      x, y, z = place_pos[0], place_pos[1], self.env.place_offset
      if self.env.isPosDistToRampValid((x, y), self.getHoldingObjType()):
        break
    y1, y2 = self.env.getY1Y2fromX(x)
    if y > y1:
      rx = self.env.ramp1_angle
      rz = self.env.ramp_rz
      d = (y - y1) * np.cos(self.env.ramp_rz)
      z += (self.env.ramp1_height + np.tan(self.env.ramp1_angle) * d)
    elif y < y2:
      rx = self.env.ramp2_angle
      rz = self.env.ramp_rz
      d = (y2 - y) * np.cos(self.env.ramp_rz)
      z += (self.env.ramp2_height + np.tan(-self.env.ramp2_angle) * d)
    else:
      rx = 0
      rz = np.random.random() * np.pi * 2

    T = transformations.euler_matrix(rz, 0, rx)
    T_random = transformations.euler_matrix(np.random.random()*np.pi, 0, 0)
    T = T_random.dot(T)
    rz, ry, rx = transformations.euler_from_matrix(T)
    # if np.abs(ry) > np.pi/6:
    #   print('ry: {}'.format(ry))
    # if np.abs(rx) > np.pi/6:
    #   print('rx: {}'.format(rx))

    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, ry, rx))

  def placeOnRamp(self, padding_dist, min_dist):
    existing_pos = [o.getXYPosition() for o in list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))]
    while True:
      try:
        place_pos = self.getValidPositions(padding_dist, min_dist, existing_pos, 1)[0]
      except NoValidPositionException:
        place_pos = self.getValidPositions(padding_dist, min_dist, [], 1)[0]
      x, y, z = place_pos[0], place_pos[1], self.env.place_offset
      y1, y2 = self.env.getY1Y2fromX(x)
      if y > y1:
        rx = -self.env.ramp1_angle
        rz = self.env.ramp_rz
        d = (y - y1) * np.cos(self.env.ramp_rz)
        z += np.tan(self.env.ramp1_angle) * d
        break
      elif y < y2:
        rx = -self.env.ramp2_angle
        rz = self.env.ramp_rz
        d = (y2 - y) * np.cos(self.env.ramp_rz)
        z += np.tan(-self.env.ramp2_angle) * d
        break
      else:
        continue
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, (rz, rx))

  def getPickingAction(self):
    if self.env.checkStructure():
      self.objs_to_remove = [o for o in self.env.structure_objs]
    if not self.objs_to_remove:
      return self.pickTallestObjOnTop()
    return self.pickTallestObjOnTop(self.objs_to_remove)

  def getPlacingAction(self):
    # if len(self.objs_to_remove) == 0:
    #   return self.placeOnTilt(self.env.max_block_size * 2, self.env.max_block_size * 2.7)
    return self.placeOnGround(self.env.max_block_size * 2, self.env.max_block_size * 2.7)
