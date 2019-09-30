import sys
sys.path.append('..')

import numpy as np
from scipy import ndimage
import numpy.random as npr

import constants
from cube import Cube
from cylinder import Cylinder

def generateObject(object_type_id, heightmap, pos, rot, size):
  if object_type_id == constants.CUBE:
    return generateCube(heightmap, pos, rot, size)
  elif object_type_id == constants.CYLINDER:
    return generateCube(heightmap, pos, rot, size)
  else:
    raise ValueError('Invalid object type ID given')

def generateCube(heightmap, pos, rot, size):
  ''''''
  cube = Cube(pos, rot, size, heightmap)
  return cube, cube.addToHeightmap(heightmap)

def generateCylinder(heightmap, pos, rot, size):
  circle = Cylinder(pos, rot, size, heightmap)
  return circle, circle.addToHeightmap(heightmap)
