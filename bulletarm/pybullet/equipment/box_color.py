import os
import pybullet as pb
import numpy as np

import bulletarm
from bulletarm.pybullet.utils import pybullet_util

class BoxColor:
  def __init__(self):
    self.root_dir = os.path.dirname(bulletarm.__file__)
    self.id = None
    self.size = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), size=(0.2, 0.2, 0.2), color=[1, 1, 1, 1]):
    bottom_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, 0.002], rgbaColor=color)
    bottom_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, 0.002])

    front_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.002, size[1]/2, size[2]/2], rgbaColor=color)
    front_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.002, size[1]/2, size[2]/2])

    back_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.002, size[1] / 2, size[2] / 2], rgbaColor=color)
    back_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.002, size[1] / 2, size[2] / 2])

    left_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/2, 0.002, size[2] / 2], rgbaColor=color)
    left_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/2, 0.002, size[2] / 2])

    right_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/2, 0.002, size[2]/2], rgbaColor=color)
    right_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/2, 0.002, size[2]/2])

    self.size = size
    self.id = pb.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=bottom_collision,
                                 baseVisualShapeIndex=bottom_visual,
                                 basePosition=pos,
                                 baseOrientation=rot,
                                 linkMasses=[1, 1, 1, 1],
                                 linkCollisionShapeIndices=[front_collision, back_collision, left_collision, right_collision],
                                 linkVisualShapeIndices=[front_visual, back_visual, left_visual, right_visual],
                                 linkPositions=[[-size[0]/2, 0, size[2]/2],
                                                [size[0]/2, 0, size[2]/2],
                                                [0, -size[1]/2, size[2]/2],
                                                [0, size[1]/2, size[2]/2]],
                                 linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                                 linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                 linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                                 linkParentIndices=[0, 0, 0, 0],
                                 linkJointTypes=[pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED],
                                 linkJointAxis=[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    )

    pb.changeDynamics(self.id,
                      -1,
                      rollingFriction=1,
                      linearDamping=0.1)

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    self.id = None
