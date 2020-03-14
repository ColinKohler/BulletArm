import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os
import glob

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants

class RandomObject(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_pattern = os.path.join(root_dir, constants.URDF_PATH, 'random_urdfs/*/*.urdf')
    found_object_directories = glob.glob(urdf_pattern)
    total_num_objects = len(found_object_directories)
    urdf_filepath = found_object_directories[np.random.choice(np.arange(total_num_objects), 1)[0]]
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(RandomObject, self).__init__(constants.CUBE, object_id)

    self.block_original_size = 0.05
    self.block_size = 0.05 * scale