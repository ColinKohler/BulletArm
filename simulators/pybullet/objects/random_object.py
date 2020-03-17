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
      found_object_directories = list(filter(lambda x: re.search(r'(000|001|002|005|006|009|011|014|020|023|025|027|032'
                                                                 r'|034|038|039|052|057|058|059|066|069|070|074|075|078'
                                                                 r'|080|081|083|089|090|091|104|110|114|115|116|117|118'
                                                                 r'|119|122|126|130|131|132|133|137|142|143|145|147|148'
                                                                 r'|149|154|157|159|161|163|166|174|176|182|183|184|185'
                                                                 r'|187|190|191|196|199|200)_2\.urdf', x),
                                             found_object_directories))
    else:
      found_object_directories = list(filter(lambda x: re.search(r'(000|001|002|005|006|009|011|014|020|023|025|027|032'
                                                                 r'|034|038|039|052|057|058|059|066|069|070|074|075|078'
                                                                 r'|080|081|083|089|090|091|104|110|114|115|116|117|118'
                                                                 r'|119|122|126|130|131|132|133|137|142|143|145|147|148'
                                                                 r'|149|154|157|159|161|163|166|174|176|182|183|184|185'
                                                                 r'|187|190|191|196|199|200)\.urdf', x),
                                             found_object_directories))
    total_num_objects = len(found_object_directories)
    urdf_filepath = found_object_directories[np.random.choice(np.arange(total_num_objects), 1)[0]]
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(RandomObject, self).__init__(constants.RANDOM, object_id)

    self.block_original_size = 0.05
    self.block_size = 0.05 * scale