import numpy as np
import numpy.random as npr

#=================================================================================================#
#                                           Objects                                               #
#=================================================================================================#

class Cube(object):
  def __init__(self, pos, rot, size):
    self.pos = pos
    self.rot = rot
    self.size = size

    self.x_min, self.x_max = int(pos[0]-size/2), int(pos[0]+size/2)
    self.y_min, self.y_max = int(pos[1]-size/2), int(pos[1]+size/2)

  def addToHeightmap(self, heightmap):
    heightmap[self.y_min:self.y_max, self.x_min:self.x_max] += self.size
    return heightmap

  def removeFromHeightmap(self, heightmap):
    heightmap[self.y_min:self.y_max, self.x_min:self.x_max] -= self.size
    return heightmap

  def isGraspValid(self, grasp_pos, grasp_rot):
    return np.allclose(grasp_pos, self.pos[:-1], atol=2*(self.size/4))

#=================================================================================================#
#                                         Generation                                              #
#=================================================================================================#

def generateCube(heightmap, pos, rot, size):
  ''''''
  cube = Cube(pos, rot, size)
  return cube, cube.addToHeightmap(heightmap)
