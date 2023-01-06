import os
import pybullet as pb
import numpy as np

import bulletarm
from bulletarm.pybullet.utils import constants


class Cabinet:
  def __init__(self):
    self.id = None

  def initialize(self, pos=(0, 0, 0), rot=(0, 0, 0, 1)):
    cabinet_urdf_filepath = os.path.join(constants.URDF_PATH, 'cabinet/cabinet.urdf')
    self.id = pb.loadURDF(cabinet_urdf_filepath, pos, rot, globalScaling=0.4)

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)
    pb.resetJointState(self.id, 1, 0)
    for i in range(50):
      pb.stepSimulation()
    pb.resetJointState(self.id, 1, 0)
    for i in range(50):
      pb.stepSimulation()
    pass

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    self.id = None

  def getLeftHandlePos(self):
    link_state = pb.getLinkState(self.id, 4)
    pos = list(link_state[0])
    pos[0] -= 0.02
    rot = list(link_state[1])
    return pos

  def getLeftHandleRot(self):
    link_state = pb.getLinkState(self.id, 4)
    pos = list(link_state[0])
    rot = list(link_state[1])
    return rot

  def isOpen(self):
    return False
