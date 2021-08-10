import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.pybullet.utils import transformations

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
    super(FlatBlock, self).__init__(constants.FLAT_BLOCK, object_id)