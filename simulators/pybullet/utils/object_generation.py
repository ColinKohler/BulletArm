import os

import pybullet as pb
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.cube import Cube
from helping_hands_rl_envs.simulators.pybullet.objects.brick import Brick
from helping_hands_rl_envs.simulators.pybullet.objects.triangle import Triangle
from helping_hands_rl_envs.simulators.pybullet.objects.roof import Roof
from helping_hands_rl_envs.simulators.pybullet.objects.cylinder import Cylinder
from helping_hands_rl_envs.simulators.pybullet.objects.random_object import RandomObject

def generateCube(pos, rot, scale):
  ''''''
  return Cube(pos, rot, scale)

def generateBrick(pos, rot, scale):
  return Brick(pos, rot, scale)

def generateTriangle(pos, rot, scale):
  # root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
  # brick_urdf_filepath = os.path.join(root_dir, 'simulators/urdf/object/triangle.urdf')
  # return pb.loadURDF(brick_urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)
  return Triangle(pos, rot, scale)

def generateRoof(pos, rot, scale):
  # root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
  # roof_urdf_filepath = os.path.join(root_dir, 'simulators/urdf/object/roof.urdf')
  # return pb.loadURDF(roof_urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)
  return Roof(pos, rot, scale)

def generateCylinder(pos, rot, scale):
  return Cylinder(pos, rot, scale)

def generateRandomObj(pos, rot, scale, z_scale=1):
  return RandomObject(pos, rot, scale, z_scale)
