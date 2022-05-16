import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import bulletarm
from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils import transformations

class Brick(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(bulletarm.__file__)
    urdf_filepath = os.path.join(root_dir, constants.OBJECTS_PATH, 'brick_small.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)
    pb.changeVisualShape(object_id, -1, rgbaColor=[0, 0, 1, 1])
    super(Brick, self).__init__(constants.BRICK, object_id)

    self.original_size = 0.05
    self.size = 0.05 * scale

  def getHeight(self):
    return self.size

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
