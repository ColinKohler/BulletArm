import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.pybullet.utils import transformations

class FlatBlock(PybulletObject):
  def __init__(self, pos, rot, scale):
    bottom_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.05*scale, 0.05*scale, 0.025], rgbaColor=[1, 1, 1, 1])
    bottom_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.05*scale, 0.05*scale, 0.025])
    object_id = pb.createMultiBody(baseMass=0.5,
                            baseCollisionShapeIndex=bottom_collision,
                            baseVisualShapeIndex=bottom_visual,
                            basePosition=pos,
                            baseOrientation=rot,
                            )
    pb.changeVisualShape(object_id, -1, rgbaColor=[0, 0, 1, 1])
    super(FlatBlock, self).__init__(constants.FLAT_BLOCK, object_id)
