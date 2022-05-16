import os
import pybullet as pb
import numpy as np

import bulletarm
from bulletarm.pybullet.utils import pybullet_util
from bulletarm.pybullet.utils import constants

class Shelf:
  def __init__(self):
    self.id = None
    self.middle_wall_id = 5
    self.target1_id = 6

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1)):
    urdf_filepath = os.path.join(constants.URDF_PATH, 'shelf.urdf')
    self.id = pb.loadURDF(urdf_filepath, pos, rot)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    self.id = None

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)
    for i in range(10):
      pb.stepSimulation()

  def getTarget1Pos(self):
    return pb.getLinkState(self.id, self.target1_id)[0]

  def isObjectOnTarget1(self, obj):
    target1_pos = self.getTarget1Pos()
    if np.linalg.norm(np.array(obj.getPosition()) - np.array(target1_pos)) > 0.05:
      return False
    contact_points = pb.getContactPoints(bodyA=self.id, linkIndexA=self.middle_wall_id)
    for p in contact_points:
      if p[2] == obj.object_id:
        return True
    return False
