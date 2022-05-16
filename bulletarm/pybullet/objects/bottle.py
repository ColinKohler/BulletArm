import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants


class Bottle(PybulletObject):
  def __init__(self, pos, rot, scale):
    self.scale = scale
    root_dir = os.path.dirname(bulletarm.__file__)
    # self.model_id = 1
    self.model_id = np.random.choice([1, 3, 4, 5, 7, 8, 9, 10])
    urdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'bottle/bottle{}.urdf'.format(self.model_id))
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
