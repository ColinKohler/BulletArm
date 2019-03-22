import numpy as np
from scipy import ndimage
import numpy.random as npr

#=================================================================================================#
#                                           Objects                                               #
#=================================================================================================#

class Cube(object):
  def __init__(self, pos, rot, size):
    self.pos = pos
    self.rot = rot
    self.size = size

    self.mask = None

    self.x_min, self.x_max = int(pos[0]-size/2), int(pos[0]+size/2)
    self.y_min, self.y_max = int(pos[1]-size/2), int(pos[1]+size/2)

  def addToHeightmap(self, heightmap):
    self.mask = np.zeros_like(heightmap, dtype=np.int)
    self.mask[self.y_min:self.y_max, self.x_min:self.x_max] = 1
    self.mask = rotateImage(self.mask, np.rad2deg(self.rot), (self.pos[1], self.pos[0]))
    self.mask = (self.mask == 1)
    heightmap[self.mask] += self.size
    return heightmap

  def removeFromHeightmap(self, heightmap):
    heightmap[self.mask] -= self.size
    return heightmap

  def isGraspValid(self, grasp_pos, grasp_rot):
    if grasp_rot > np.pi:
      grasp_rot -= np.pi
    valid_rot1 = self.rot
    if valid_rot1 < np.pi/2:
      valid_rot2 = valid_rot1 + np.pi/2
    else:
      valid_rot2 = valid_rot1 - np.pi/2
    return np.allclose(grasp_pos[:-1], self.pos[:-1], atol=(self.size/2)) and \
           (np.abs(grasp_rot-valid_rot1) < np.pi/8 or np.abs(grasp_rot-valid_rot2) < np.pi/8) and \
           grasp_pos[-1] < self.pos[-1]

#=================================================================================================#
#                                         Generation                                              #
#=================================================================================================#

def generateCube(heightmap, pos, rot, size):
  ''''''
  cube = Cube(pos, rot, size)
  return cube, cube.addToHeightmap(heightmap)

def rotateImage(img, angle, pivot):
  padX = [img.shape[1] - pivot[1], pivot[1]]
  padY = [img.shape[0] - pivot[0], pivot[0]]
  imgP = np.pad(img, [padY, padX], 'constant')
  imgR = ndimage.rotate(imgP, angle, reshape=False)
  return imgR[padY[0]: -padY[1], padX[0]: -padX[1]]