import os
import pybullet as pb
import numpy as np

from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.utils import constants

class TeapotLid(PybulletObject):
  def __init__(self, pos, rot, scale, model_id=1):
    self.teapot_model_id = model_id
    urdf_filepath = os.path.join(constants.URDF_PATH, 'teapot/{}/lid.urdf'.format(self.teapot_model_id))
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)
    super().__init__(constants.TEAPOT, object_id)
    self.scale = scale

  def getGraspPosition(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    T = transformations.quaternion_matrix(rot)
    pos = np.array(pos)
    if self.teapot_model_id in [1, 2]:
      pos += (T[:3, 2]*0.25*self.scale)
    elif self.teapot_model_id in [3]:
      pos += (T[:3, 2]*0.1*self.scale)
    elif self.teapot_model_id in [4]:
      pos += (T[:3, 2]*0.2*self.scale)
    elif self.teapot_model_id in [5]:
      pos += (T[:3, 2]*0.3*self.scale)

    return pos

  def getPosition(self):
    return self.getGraspPosition()
