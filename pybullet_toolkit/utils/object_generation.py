import os

import pybullet as pb
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs

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
  return rot

def getObjectPose(obj):
  pos, rot = pb.getBasePositionAndOrientation(obj)
  return pos, rot

def generateBrick(pos, rot, scale):
  root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
  brick_urdf_filepath = os.path.join(root_dir, 'urdf/object/brick_small.urdf')
  return pb.loadURDF(brick_urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

def generateTriangle(pos, rot, scale):
  root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
  brick_urdf_filepath = os.path.join(root_dir, 'urdf/object/0.urdf')
  return pb.loadURDF(brick_urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)