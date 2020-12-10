import os
import pybullet as pb
import numpy as np

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators.pybullet.utils import transformations
from helping_hands_rl_envs.simulators import constants

class TeapotBase(PybulletObject):
  def __init__(self, pos, rot, scale):
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    urdf_filepath = os.path.join(root_dir, 'simulators/urdf/teapot/base.urdf')
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)
    super().__init__(constants.TEAPOT, object_id)
    self.scale = scale

  def getHandlePos(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    T = transformations.quaternion_matrix(rot)
    pos = np.array(pos)
    pos += (T[:3, 1]*0.75*self.scale)
    pos += (T[:3, 2]*0.6*self.scale)
    return pos

  def getPosition(self):
    return self.getHandlePos()

  def getOpenPos(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    pos = np.array(pos)
    pos[2] += 1*self.scale
    return pos