import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants

class Pallet(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'pallet/pallet3.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale, useFixedBase=True)

    super(Pallet, self).__init__(constants.PALLET, object_id)
