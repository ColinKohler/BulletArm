import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants

class Cup(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    # urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'kitchenUtensils/urdf/glass2.urdf')
    urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'cup/cup.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(Cup, self).__init__(constants.CUBE, object_id)
