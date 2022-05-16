import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations

class Box(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(bulletarm.__file__)
    urdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'box/box.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale, flags=pb.URDF_ENABLE_SLEEPING)
    shade = np.random.rand() + 0.5
    color = np.float32([shade * 156, shade * 117, shade * 95, 255]) / 255
    pb.changeVisualShape(object_id, -1, rgbaColor=color)
    super(Box, self).__init__(constants.BOX, object_id)

  def getRotation(self):
    pos, rot = self.getPose()
    return rot

  def getPose(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    T = transformations.quaternion_matrix(rot)
    t = 0
    while T[2, 2] < 0.5 and t < 3:
      T = T.dot(transformations.euler_matrix(0, np.pi/2, 0))
      t += 1
    rot = transformations.quaternion_from_matrix(T)
    return list(pos), list(rot)
