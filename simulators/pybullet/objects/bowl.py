import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants

class Bowl(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'bowl/bowl.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(Bowl, self).__init__(constants.CUBE, object_id)

  def getGraspPose(self):
    link_state = pb.getLinkState(self.object_id, 0)
    pos = link_state[0]
    rot = link_state[1]
    return list(pos), list(rot)

  # def getRotation(self):
  #   link_state = pb.getLinkState(self.object_id, 0)
  #   rot = link_state[1]
  #   return list(rot)

  # def getPose(self):
  #   link_state = pb.getLinkState(self.object_id, 0)
  #   pos = link_state[0]
  #   rot = link_state[1]
  #   return list(pos), list(rot)