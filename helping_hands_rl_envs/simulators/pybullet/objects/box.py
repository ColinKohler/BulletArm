import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants

class Box(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'box/box.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale, flags=pb.URDF_ENABLE_SLEEPING)
    shade = np.random.rand() + 0.5
    color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
    pb.changeVisualShape(object_id, -1, rgbaColor=color)
    super(Box, self).__init__(constants.BOX, object_id)
