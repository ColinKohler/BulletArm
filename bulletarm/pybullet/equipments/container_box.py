import os
import pybullet as pb
import numpy as np

import bulletarm
from bulletarm.pybullet.utils import pybullet_util

class ContainerBox:
  def __init__(self):
    self.root_dir = os.path.dirname(bulletarm.__file__)
    self.id = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), size=(0.2, 0.2, 0.2), thickness=0.005):
    bottom_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, thickness], rgbaColor=[1, 1, 1, 1])
    bottom_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, thickness])

    front_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[thickness, size[1]/2, size[2]/2], rgbaColor=[1, 1, 1, 1])
    front_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[thickness, size[1]/2, size[2]/2])

    back_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[thickness, size[1] / 2, size[2] / 2], rgbaColor=[1, 1, 1, 1])
    back_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[thickness, size[1] / 2, size[2] / 2])

    left_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/2, thickness, size[2] / 2], rgbaColor=[1, 1, 1, 1])
    left_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/2, thickness, size[2] / 2])

    right_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/2, thickness, size[2]/2], rgbaColor=[1, 1, 1, 1])
    right_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/2, thickness, size[2]/2])

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

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    self.id = None
