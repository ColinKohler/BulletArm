import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants


class SquarePeg(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(bulletarm.__file__)
    urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'peg_insertion/peg/Peg.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)
    super().__init__(constants.SQUARE_PEG, object_id)

    self.original_height = 0.025
    self.original_size = 0.05

    self.height = self.original_height * scale
    self.size = self.original_size * scale

  def getHeight(self):
    return self.height
