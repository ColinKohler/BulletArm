import os
import pybullet as pb
import numpy as np

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util

class Drawer:
  def __init__(self):
    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    self.id = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1)):
    drawer_urdf_filepath = os.path.join(self.root_dir, 'simulators/urdf/drawer.urdf')
    self.id = pb.loadURDF(drawer_urdf_filepath, pos, rot, globalScaling=0.5)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    self.id = None

  def getHandlePosition(self):
    return pb.getLinkState(self.id, 9)[4]

  def reset(self):
    pb.resetJointState(self.id, 1, 0)

  def isDrawerOpen(self):
    return pb.getJointState(self.id, 1)[0] > 0.15

  def isDrawerClosed(self):
    return pb.getJointState(self.id, 1)[0] < 0.02
