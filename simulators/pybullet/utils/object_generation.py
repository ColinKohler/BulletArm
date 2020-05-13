import os

import pybullet as pb
import numpy as np
import numpy.random as npr

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.cube import Cube
from helping_hands_rl_envs.simulators.pybullet.objects.cylinder import Cylinder
from helping_hands_rl_envs.simulators.pybullet.objects.brick import Brick
from helping_hands_rl_envs.simulators.pybullet.objects.triangle import Triangle
from helping_hands_rl_envs.simulators.pybullet.objects.roof import Roof

def generateCube(pos, rot, scale):
  ''''''
  return Cube(pos, rot, scale)

def generateBrick(pos, rot, scale):
  return Brick(pos, rot, scale)

def generateCylinder(pos, rot, scale):
  return Cylinder(pos, rot, scale)

def generateTriangle(pos, rot, scale):
  return Triangle(pos, rot, scale)

def generateRoof(pos, rot, scale):
  return Roof(pos, rot, scale)
