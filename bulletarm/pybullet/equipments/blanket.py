import os
import pybullet as pb
import numpy as np

import bulletarm
from bulletarm.pybullet.utils import pybullet_util

class Blanket:
  def __init__(self):
    self.root_dir = os.path.dirname(bulletarm.__file__)
    self.id = None
    self.object_id = self.id

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), size=(0.2, 0.2, 0.2)):
    bottom_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2], rgbaColor=[1, 1, 1, 1])
    bottom_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, size[2]/2])

    self.id = pb.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=bottom_collision,
                                 baseVisualShapeIndex=bottom_visual,
                                 basePosition=pos,
                                 baseOrientation=rot,
    )
    self.object_id = self.id

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    self.id = None
    self.object_id = self.id

