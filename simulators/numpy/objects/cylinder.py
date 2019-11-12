import sys
sys.path.append('..')

import numpy as np

from helping_hands_rl_envs.simulators.numpy.objects.numpy_object import NumpyObject
from helping_hands_rl_envs.simulators import constants

class Cylinder(NumpyObject):
  def __init__(self, pos, rot, size, heightmap):
    super(Cylinder, self).__init__(constants.CYLINDER, pos, rot, size, heightmap)

    self.radius = size/2
    self.mask = self.getMask()
    self.chunk_before = None
    self.on_top = True

  def addToHeightmap(self, heightmap, pos=None, rot=None):
    if pos is not None:
      self.pos = list(map(int, pos))
      self.getMask()
      base_h = heightmap[self.mask].max() if self.mask.sum() > 0 else 0
      self.pos[-1] = self.height + base_h

    self.chunk_before = heightmap[self.mask]
    heightmap[self.mask] = self.pos[-1]

    return heightmap

  def removeFromHeightmap(self, heightmap):
    heightmap[self.mask] = self.chunk_before
    return heightmap

  def isGraspValid(self, grasp_pos, grasp_rot, check_rot=True):
    return np.allclose(grasp_pos[:-1], self.pos[:-1], atol=(self.size/2)) and \
           grasp_pos[-1] < self.pos[-1] and \
           self.on_top

  def isStackValid(self, stack_pos, stack_rot, bottom_object, check_rot=False):
    if bottom_object == self or not bottom_object.on_top or type(bottom_object) is not Cylinder:
      return False
    if np.allclose(stack_pos[:-1], bottom_object.pos[:-1], atol=(bottom_object.size / 2)) and \
        bottom_object.pos[-1]<=stack_pos[-1]:
      return True
    return False

  def getMask(self):
    y, x = np.ogrid[-self.pos[0]:self.heightmap_size-self.pos[0], -self.pos[1]:self.heightmap_size-self.pos[1]]
    region = x*x + y*y <= self.radius*self.radius
    mask = np.zeros((self.heightmap_size, self.heightmap_size), dtype=np.int)
    mask[region.T] = 1
    mask = (mask == 1)

    return mask
