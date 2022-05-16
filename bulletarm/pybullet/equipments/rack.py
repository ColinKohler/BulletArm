import os
import pybullet as pb
import numpy as np

import bulletarm
from bulletarm.pybullet.utils import pybullet_util
from bulletarm.pybullet.utils import transformations
from bulletarm.pybullet.utils import constants

class Rack:
  def __init__(self, n=3, dist=0.1):
    self.n = n
    self.ids = []
    self.dist = dist

  def getEachPos(self, pos=(0,0,0), rot=(0,0,0,1)):
    poss = []
    base_x, base_y, base_z = pos
    rx, ry, rz = transformations.euler_from_quaternion(rot)
    for i in range(self.n):
      x = base_x + i * self.dist * np.cos(rz)
      y = base_y + i * self.dist * np.sin(rz)
      z = base_z
      poss.append((x, y, z))
    return poss

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), fixed=False):
    urdf_filepath = os.path.join(constants.URDF_PATH, 'rack2.urdf')
    poss = self.getEachPos(pos, rot)
    for i in range(self.n):
      self.ids.append(pb.loadURDF(urdf_filepath, poss[i], rot, useFixedBase=fixed))
      if len(self.ids) > 1:
        pb.createConstraint(self.ids[-2], -1, self.ids[-1], -1,
                            jointType=pb.JOINT_FIXED, jointAxis=[0, 0, 0],
                            parentFramePosition=[self.dist, 0, 0],
                            childFramePosition=[0, 0, 0],
                            childFrameOrientation=pb.getQuaternionFromEuler([0, 0, 0]))

    # base_x, base_y, base_z = pos
    # rx, ry, rz = transformations.euler_from_quaternion(rot)
    # for i in range(self.n):
    #   x = base_x + i * self.dist * np.cos(rz)
    #   y = base_y + i * self.dist * np.sin(rz)
    #   z = base_z
    #   self.ids.append(pb.loadURDF(urdf_filepath, (x, y, z), rot))

  def remove(self):
    for idx in self.ids:
      pb.removeBody(idx)
    self.ids = []

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    poss = self.getEachPos(pos, rot)
    for i in range(self.n):
      pb.resetBasePositionAndOrientation(self.ids[i], poss[i], rot)
    for i in range(10):
      pb.stepSimulation()

  def getObjInitPosList(self):
    poss = []
    for idx in self.ids:
      poss.append(pb.getLinkState(idx, 2)[0])
    return poss[1:]

  def getObjInitRotList(self):
    rots = []
    for idx in self.ids:
      rots.append(pb.getLinkState(idx, 2)[1])
    return rots[1:]
