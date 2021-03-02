import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

BOWL1_OFFSET=0.09

def getBowlRadius(model_id):
  if model_id == 1:
    return 0.09
  elif model_id == 2:
    return 0.09
  elif model_id == 3:
    return 0.09
  else:
    raise NotImplementedError

def getZOffset(model_id):
  if model_id == 1:
    return 0.01
  elif model_id == 2:
    return 0.02
  elif model_id == 3:
    return 0.0
  else:
    raise NotImplementedError

class Bowl(PybulletObject):
  def __init__(self, pos, rot, scale):
    self.scale = scale
    root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    self.model_id = 1
    urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'bowl/bowl{}.urdf'.format(self.model_id))
    object_id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=scale)

    super(Bowl, self).__init__(constants.CUBE, object_id)

  def getGraspPose(self):
    pos, rot_q = self.getPose()
    rot_e = transformations.euler_from_quaternion(rot_q)
    x, y, z = pos
    rx, ry, rz = rot_e
    theta = np.random.random() * 2 * np.pi
    rz = rz + theta
    ry += np.pi/10
    dx = np.cos(rz) * getBowlRadius(self.model_id) * self.scale
    dy = np.sin(rz) * getBowlRadius(self.model_id) * self.scale
    x += dx
    y += dy
    z += getZOffset(self.model_id)
    rot_q = transformations.quaternion_from_euler(rx, ry, rz)

    return [x, y, z], rot_q



  # def getRotation(self):
  #   link_state = pb.getLinkState(self.object_id, 0)
  #   rot = link_state[1]
  #   return list(rot)

  # def getPose(self):
  #   link_state = pb.getLinkState(self.object_id, 0)
  #   pos = link_state[0]
  #   rot = link_state[1]
  #   return list(pos), list(rot)