import os
import pybullet as pb
import numpy as np

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.utils import pybullet_util
from helping_hands_rl_envs.simulators.pybullet.equipments.drawer_handle import DrawerHandle

class Drawer:
  def __init__(self):
    self.root_dir = os.path.dirname(helping_hands_rl_envs.__file__)
    self.id = None
    self.handle = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1)):
    drawer_urdf_filepath = os.path.join(self.root_dir, 'simulators/urdf/drawer.urdf')
    self.id = pb.loadURDF(drawer_urdf_filepath, pos, rot, globalScaling=0.5)
    self.handle = DrawerHandle(self.id)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    if self.handle:
      pb.removeBody(self.handle.id)
    self.id = None
    self.handle = None

  def isObjInsideDrawer(self, obj):
    contact_points = obj.getContactPoints()
    for p in contact_points:
      if p[2] == self.id and p[4] == 4:
        return True
    return False

  def getHandlePosition(self):
    # return pb.getLinkState(self.handle.id, 0)[4]
    return self.handle.getPosition()

  def getHandleRotation(self):
    return self.handle.getRotation()

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)
    for i in range(50):
      pb.stepSimulation()
    pb.resetJointState(self.id, 1, 0)
    for i in range(50):
      pb.stepSimulation()
    pass

  def isDrawerOpen(self):
    return pb.getJointState(self.id, 1)[0] > 0.15

  def isDrawerClosed(self):
    return pb.getJointState(self.id, 1)[0] < 0.02

  def getObjInitPos(self):
    return pb.getLinkState(self.id, 5)[0]