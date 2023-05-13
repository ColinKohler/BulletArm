import os
import pybullet as pb
import numpy as np

import bulletarm
from bulletarm.pybullet.utils import constants

class RoundPegHole(object):
  '''

  '''
  def __init__(self):
    self.id = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), size=(0.3, 0.3, 0.1)):
    ''''''
    root_dir = os.path.dirname(bulletarm.__file__)
    urdf_filepath = os.path.join(root_dir, constants.URDF_PATH, 'peg_insertion/round_peg/fixture/Hole.urdf')
    self.id = pb.loadURDF(urdf_filepath, basePosition=pos, baseOrientation=rot, globalScaling=1.0, useFixedBase=True)

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    ''''''
    pb.resetBasePositionAndOrientation(self.id, pos, rot)

  def getPose(self):
    ''''''
    pos, rot = pb.getBasePositionAndOrientation(self.id)
    return list(pos), list(rot)

  def getHolePose(self):
    pos, rot = pb.getBasePositionAndOrientation(self.id)
    return list(pos), list(rot)

  def remove(self):
    ''''''
    if self.id:
      pb.removeBody(self.id)
    self.id = None
