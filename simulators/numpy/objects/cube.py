import sys
sys.path.append('..')

import numpy as np

from helping_hands_rl_envs.simulators.numpy.objects.numpy_object import NumpyObject
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.numpy import utils

class Cube(NumpyObject):
  def __init__(self, pos, rot, size, heightmap):
    super(Cube, self).__init__(constants.CUBE, pos, rot, size)

    self.x_min, self.x_max = max(0, int(pos[0] - size / 2)), min(heightmap.shape[0], int(pos[0] + size / 2))
    self.y_min, self.y_max = max(0, int(pos[1] - size / 2)), min(heightmap.shape[1], int(pos[1] + size / 2))
    self.mask = self.getMask(heightmap)

    self.chunk_before = None
    self.on_top = True

  def addToHeightmap(self, heightmap, pos=None, rot=None):
    if rot is not None:
      self.rot += rot
      while self.rot > np.pi:
        self.rot -= np.pi
    if pos is not None:
      self.pos = list(map(int, pos))
      self.x_min, self.x_max = max(0, int(pos[0] - self.size / 2)), min(heightmap.shape[0], int(pos[0] + self.size / 2))
      self.y_min, self.y_max = max(0, int(pos[1] - self.size / 2)), min(heightmap.shape[1], int(pos[1] + self.size / 2))
      self.mask = np.zeros_like(heightmap, dtype=np.int)
      self.mask[self.y_min:self.y_max, self.x_min:self.x_max] = 1
      self.mask = utils.rotateImage(self.mask, np.rad2deg(self.rot), (self.pos[1], self.pos[0]))
      self.mask = (self.mask == 1)
      base_h = heightmap[self.mask].max() if self.mask.sum() > 0 else 0
      self.pos[-1] = self.height + base_h

    self.chunk_before = heightmap[self.mask]
    heightmap[self.mask] = self.pos[-1]

    return heightmap

  def removeFromHeightmap(self, heightmap):
    heightmap[self.mask] = self.chunk_before
    return heightmap

  def isGraspValid(self, grasp_pos, grasp_rot, check_rot=True):
    # check position, height, and on top
    if np.allclose(grasp_pos[:-1], self.pos[:-1], atol=(self.size/2)) and \
        grasp_pos[-1] < self.pos[-1] and self.on_top:
      if not check_rot:
        return True
      else:
        # check rotation
        obj_rot = self.rot
        angle = np.pi - np.abs(np.abs(grasp_rot - obj_rot) - np.pi)
        while angle > np.pi / 2:
          angle -= np.pi / 2
        angle = min(angle, np.pi / 2 - angle)
        return angle < np.pi / 12
    return False

  def isStackValid(self, stack_pos, stack_rot, bottom_block, check_rot=False):
    # check bottom block on top, type
    if bottom_block == self or not bottom_block.on_top or type(bottom_block) is not Cube:
      return False
    # check position, height
    if np.allclose(stack_pos[:-1], bottom_block.pos[:-1], atol=(bottom_block.size / 4)) and \
        bottom_block.pos[-1]<=stack_pos[-1]:
      if not check_rot:
        return True
      else:
        # check rot
        stack_rot += self.rot
        while stack_rot > np.pi:
          stack_rot -= np.pi
        valid_rot1 = bottom_block.rot
        if valid_rot1 < np.pi/2:
          valid_rot2 = valid_rot1 + np.pi/2
        else:
          valid_rot2 = valid_rot1 - np.pi/2
        valid_rot3 = valid_rot1 + np.pi
        valid_rot4 = valid_rot2 + np.pi
        valid_rots = np.array([valid_rot1, valid_rot2, valid_rot3, valid_rot4])
        angle = np.pi - np.abs(np.abs(valid_rots - stack_rot) - np.pi)
        return np.any(angle < np.pi/7)
    return False

  def getMask(self, heightmap):
    mask = np.zeros_like(heightmap, dtype=np.int)
    mask[self.y_min:self.y_max, self.x_min:self.x_max] = 1
    mask = utils.rotateImage(mask, np.rad2deg(self.rot), (self.pos[1], self.pos[0]))
    mask = (mask == 1)

    return mask
