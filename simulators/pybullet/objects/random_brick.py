import sys
sys.path.append('..')

import pybullet as pb
import numpy as np
import os

import helping_hands_rl_envs
from helping_hands_rl_envs.simulators.pybullet.objects.pybullet_object import PybulletObject
from helping_hands_rl_envs.simulators import constants

class RandomBrick(PybulletObject):
  def __init__(self, pos, rot, x_scale, y_scale, z_scale):
    visualShapeId = pb.createVisualShape(shapeType=pb.GEOM_BOX,
                                         halfExtents=[0.05/2*x_scale, 0.15/2*y_scale, 0.05/2*z_scale])
    collisionShapeId = pb.createCollisionShape(shapeType=pb.GEOM_BOX,
                                               halfExtents=[0.05/2*x_scale, 0.15/2*y_scale, 0.05/2*z_scale])
    object_id = pb.createMultiBody(baseMass=0.1,
                                   baseInertialFramePosition=[0, 0, 0],
                                   baseCollisionShapeIndex=collisionShapeId,
                                   baseVisualShapeIndex=visualShapeId,
                                   basePosition=pos,
                                   baseOrientation=rot)
    super(RandomBrick, self).__init__(constants.BRICK, object_id)
