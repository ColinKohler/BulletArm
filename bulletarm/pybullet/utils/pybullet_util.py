import pybullet as pb
import numpy as np
from bulletarm.pybullet.utils import constants

def getMatrix(pos, rot):
  T = np.eye(4)
  T[:3, :3] = np.array(pb.getMatrixFromQuaternion(rot)).reshape((3, 3))
  T[:3, 3] = pos
  return T

def getPadding(t, max_block_size):
  if t in (constants.CUBE, constants.TRIANGLE, constants.RANDOM):
    padding = max_block_size * 1.5
  elif t in (constants.BRICK, constants.ROOF):
    padding = max_block_size * 3.4
  else:
    padding = max_block_size * 1.5
  return padding

def getMinDistance(t, max_block_size):
  if t in (constants.CUBE, constants.TRIANGLE, constants.RANDOM):
    min_distance = max_block_size * 2.4
  elif t in (constants.BRICK, constants.ROOF):
    min_distance = max_block_size * 3.4
  else:
    min_distance = max_block_size * 2.4
  return min_distance
