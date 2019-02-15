import pybullet as pb
import numpy as np
import numpy.random as npr

def generateCube(pos, rot, scale):
  return pb.loadURDF('cube_small.urdf', basePosition=pos, baseOrientation=rot, globalScaling=scale)
