import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os
import glob

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants

root_dir = os.path.dirname(bulletarm.__file__)
urdf_pattern = os.path.join(root_dir, constants.OBJECTS_PATH, 'random_household_object/*/*.urdf')
found_object_directories = glob.glob(urdf_pattern)
# found_object_directories = list(filter(lambda x: re.search(r'(flashlight)\.urdf', x),
#                                        found_object_directories))
total_num_objects = len(found_object_directories)

class RandomHouseHoldObject(PybulletObject):
  def __init__(self, pos, rot, scale):
    # for i, urdf in enumerate(found_object_directories):
    #   pb.loadURDF(urdf, basePosition=[0.1+i//4*0.15, 0.1+i%4*0.15, 0.05], baseOrientation=(0, 0, 0, 1), globalScaling=scale)

    urdf_filepath = found_object_directories[np.random.choice(np.arange(total_num_objects), 1)[0]]
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(RandomHouseHoldObject, self).__init__(constants.RANDOM, object_id)
