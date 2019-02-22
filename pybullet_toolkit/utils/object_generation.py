import pybullet as pb
import numpy as np
import numpy.random as npr

def generateCube(pos, rot, scale):
  ''''''
  return pb.loadURDF('cube_small.urdf', basePosition=pos, baseOrientation=rot, globalScaling=scale)

def getObjectPosition(obj):
  ''''''
  pos, rot = pb.getBasePositionAndOrientation(obj)
  return pos

def getObjectRotation(obj):
  ''''''
  pos, rot = pb.getBasePositionAndOrientation(obj)
  return pos

def getObjectPose(obj):
  pos, rot = pb.getBasePositionAndOrientation(obj)
  return pos, rot
