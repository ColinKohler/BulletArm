import numpy as np
from scipy import ndimage
import numpy.random as npr

#=================================================================================================#
#                                           Objects                                               #
#=================================================================================================#

class Cylinder(object):
  def __init__(self, pos, rot, size, heightmap):
    self.pos = pos
    self.size = size
    self.radius = size/2
    self.height = pos[-1]
    self.heightmap_size = heightmap.shape[0]
    self.mask = None
    self.getMask()
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

  def getMask(self):
    y, x = np.ogrid[-self.pos[0]:self.heightmap_size-self.pos[0], -self.pos[1]:self.heightmap_size-self.pos[1]]
    region = x*x + y*y <= self.radius*self.radius
    self.mask = np.zeros((self.heightmap_size, self.heightmap_size), dtype=np.int)
    self.mask[region.T] = 1
    self.mask = (self.mask == 1)

  def removeFromHeightmap(self, heightmap):
    heightmap[self.mask] = self.chunk_before
    return heightmap

  def isGraspValid(self, grasp_pos, grasp_rot):
    return np.allclose(grasp_pos[:-1], self.pos[:-1], atol=(self.size/2)) and \
           grasp_pos[-1] < self.pos[-1] and \
           self.on_top

  def isStackValid(self, stack_pos, stack_rot, bottom_object):
    if bottom_object == self or not bottom_object.on_top or type(bottom_object) is not Cylinder:
      return False
    if np.allclose(stack_pos[:-1], bottom_object.pos[:-1], atol=(bottom_object.size / 2)) and \
        bottom_object.pos[-1]<=stack_pos[-1]:
      return True
    return False


class Cube(object):
  def __init__(self, pos, rot, size, heightmap):
    self.pos = pos
    self.rot = rot
    self.size = size
    self.height = pos[-1]

    self.x_min, self.x_max = max(0, int(pos[0] - size / 2)), min(heightmap.shape[0], int(pos[0] + size / 2))
    self.y_min, self.y_max = max(0, int(pos[1] - size / 2)), min(heightmap.shape[1], int(pos[1] + size / 2))
    self.mask = np.zeros_like(heightmap, dtype=np.int)
    self.mask[self.y_min:self.y_max, self.x_min:self.x_max] = 1
    self.mask = rotateImage(self.mask, np.rad2deg(self.rot), (self.pos[1], self.pos[0]))
    self.mask = (self.mask == 1)

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
      self.mask = rotateImage(self.mask, np.rad2deg(self.rot), (self.pos[1], self.pos[0]))
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
        if grasp_rot > np.pi:
          grasp_rot -= np.pi
        valid_rot1 = self.rot
        if valid_rot1 < np.pi/2:
          valid_rot2 = valid_rot1 + np.pi/2
        else:
          valid_rot2 = valid_rot1 - np.pi/2
        valid_rot3 = valid_rot1 + np.pi
        valid_rot4 = valid_rot2 + np.pi
        valid_rots = np.array([valid_rot1, valid_rot2, valid_rot3, valid_rot4])
        angle = np.pi - np.abs(np.abs(valid_rots - grasp_rot) - np.pi)
        return np.any(angle < np.pi/7)
    return False

  def isStackValid(self, stack_pos, stack_rot, bottom_block, check_rot=False):
    # check bottom block on top, type
    if bottom_block == self or not bottom_block.on_top or type(bottom_block) is not Cube:
      return False
    # check position, height
    if np.allclose(stack_pos[:-1], bottom_block.pos[:-1], atol=(bottom_block.size / 2)) and \
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

#=================================================================================================#
#                                         Generation                                              #
#=================================================================================================#

def generateCube(heightmap, pos, rot, size):
  ''''''
  cube = Cube(pos, rot, size, heightmap)
  return cube, cube.addToHeightmap(heightmap)

def generateCylinder(heightmap, pos, rot, size):
  circle = Cylinder(pos, rot, size, heightmap)
  return circle, circle.addToHeightmap(heightmap)

def rotateImage(img, angle, pivot):
  pad_x = [img.shape[1] - pivot[1], pivot[1]]
  pad_y = [img.shape[0] - pivot[0], pivot[0]]
  img_recenter = np.pad(img, [pad_y, pad_x], 'constant')
  img_2x = ndimage.zoom(img_recenter, zoom=2, order=0)
  img_r_2x = ndimage.rotate(img_2x, angle, reshape=False)
  img_r = img_r_2x[::2, ::2]
  if pad_y[1] == 0 and pad_x[1]== 0:
    result = img_r[pad_y[0]:, pad_x[0]:]
  elif pad_y[1] == 0:
    result = img_r[pad_y[0]:, pad_x[0]: -pad_x[1]]
  elif pad_x[1] == 0:
    result = img_r[pad_y[0]: -pad_y[1], pad_x[0]:]
  else:
    result = img_r[pad_y[0]: -pad_y[1], pad_x[0]: -pad_x[1]]
  return result
