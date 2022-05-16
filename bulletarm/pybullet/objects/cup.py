import sys
sys.path.append('..')

import pybullet as pb
import os

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants

class Cup(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(bulletarm.__file__)
    # urdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'kitchenUtensils/urdf/glass2.urdf')
    urdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'cup/cup.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(Cup, self).__init__(constants.CUP, object_id)

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
