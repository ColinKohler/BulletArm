import os
import pybullet as pb
import numpy as np

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators import constants

class TeapotBase(PybulletObject):
  def __init__(self, pos, rot, scale, model_id=1):
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    self.teapot_model_id = model_id
    urdf_filepath = os.path.join(root_dir, 'simulators/urdf/teapot/{}/base.urdf'.format(self.teapot_model_id))
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)
    super().__init__(constants.TEAPOT, object_id)
    self.scale = scale

  def getGraspPosition(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    T = transformations.quaternion_matrix(rot)
    pos = np.array(pos)
    if self.teapot_model_id == 1:
      pos += (T[:3, 1]*0.75*self.scale)
      pos += (T[:3, 2]*0.6*self.scale)
    elif self.teapot_model_id in [2, 3]:
      pos += (T[:3, 1]*0.7*self.scale)
      pos += (T[:3, 2]*0.55*self.scale)
    elif self.teapot_model_id in [4]:
      pos += (T[:3, 1]*0.85*self.scale)
      pos += (T[:3, 2]*0.4*self.scale)
    elif self.teapot_model_id == 5:
      pos += (T[:3, 1]*0.75*self.scale)
      pos += (T[:3, 2]*0.6*self.scale)
    return pos

  def getPosition(self):
    return self.getGraspPosition()

  def getOpenPos(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    pos = np.array(pos)
    pos[2] += 1*self.scale
    return pos