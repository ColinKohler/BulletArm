import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os
import glob
import re

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants

class RandomObject(PybulletObject):
  def __init__(self, pos, rot, scale, z_scale=1):
    assert z_scale in (1, 2)
    self.z_scale = z_scale
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_pattern = os.path.join(root_dir, constants.URDF_PATH, 'random_urdfs/*/*.urdf')
    found_object_directories = glob.glob(urdf_pattern)
    if z_scale == 2:
      found_object_directories = list(filter(lambda x: re.search(r'(002|005|027|032|034|039|059|066|070|075|080|083|091'
                                                                 r'|116|118|122|131|137|142|143|145|149|154|176|183|187'
                                                                 r'|199|200)_2\.urdf', x),
                                             found_object_directories))
    else:
      found_object_directories = list(filter(lambda x: re.search(r'(002|005|027|032|034|039|059|066|070|075|080|083|091'
                                                                 r'|116|118|122|131|137|142|143|145|149|154|176|183|187'
                                                                 r'|199|200)\.urdf', x),
                                             found_object_directories))
    total_num_objects = len(found_object_directories)
    urdf_filepath = found_object_directories[np.random.choice(np.arange(total_num_objects), 1)[0]]
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(RandomObject, self).__init__(constants.RANDOM, object_id)

    self.block_original_size = 0.05
    self.block_size = 0.05 * scale