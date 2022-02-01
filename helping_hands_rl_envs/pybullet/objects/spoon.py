import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.pybullet.utils import constants

class Spoon(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'spoon/spoon.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(Spoon, self).__init__(constants.SPOON, object_id)
