import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

def getZOffset(model_id):
  if model_id == 1:
    return 0.01
  elif model_id == 2:
    return 0.02
  elif model_id == 3:
    return 0.0
  else:
    raise NotImplementedError

class Bottle(PybulletObject):
  def __init__(self, pos, rot, scale):
    self.scale = scale
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    self.model_id = 1
    urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'bottle/bottle{}.urdf'.format(self.model_id))
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(Bottle, self).__init__(constants.BOTTLE, object_id)

  def getGraspRotation(self):
    link_state = pb.getLinkState(self.object_id, 0)
    rot_q = link_state[1]
    return list(rot_q)

  def getGraspPosition(self):
    link_state = pb.getLinkState(self.object_id, 0)
    pos = link_state[0]
    return list(pos)

  def getGraspPose(self):
    return self.getGraspPosition(), self.getGraspRotation()




  # def getRotation(self):
  #   link_state = pb.getLinkState(self.object_id, 0)
  #   rot = link_state[1]
  #   return list(rot)

  # def getPose(self):
  #   link_state = pb.getLinkState(self.object_id, 0)
  #   pos = link_state[0]
  #   rot = link_state[1]
  #   return list(pos), list(rot)