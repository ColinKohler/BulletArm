import os
import pybullet as pb
import numpy as np

import bulletarm
from bulletarm.pybullet.utils import pybullet_util

class Corner:
  def __init__(self):
    self.id = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), size=(0.2, 0.2, 0.1)):
    bottom_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[1]/2, size[1]/2, 0.001], rgbaColor=[1, 1, 1, 1])
    bottom_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[1]/2, size[1]/2, 0.001])

    front_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.01, size[1]/2, size[2]/2], rgbaColor=[1, 1, 1, 1])
    front_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0.01, size[1]/2, size[2]/2])

    left_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/2, 0.01, size[2] / 2], rgbaColor=[1, 1, 1, 1])
    left_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/2, 0.01, size[2] / 2])

    obj_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1, 0, 0, 0])
    obj_collision = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0, 0, 0], rgbaColor=[1, 0, 0, 1])

    pull_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[0, 1, 0, 0])
    pull_collision= pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0, 0, 0], rgbaColor=[0, 1, 0, 1])

    press_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[0, 0, 1, 0])
    press_collision= pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0, 0, 0], rgbaColor=[0, 0, 1, 1])

    self.id = pb.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=bottom_collision,
                                 baseVisualShapeIndex=bottom_visual,
                                 basePosition=pos,
                                 baseOrientation=rot,
                                 linkMasses=[1, 1, 1, 1, 1],
                                 linkCollisionShapeIndices=[front_collision, left_collision, obj_collision, pull_collision, press_collision],
                                 linkVisualShapeIndices=[front_visual, left_visual, obj_visual, pull_visual, press_visual],
                                 linkPositions=[[-size[0]/2, 0, size[2]/2],
                                                [0, -size[1]/2, size[2]/2],
                                                [-(size[0]/2 - 0.04), -(size[0]/2 - 0.04), 0.03],
                                                [0.06, 0.06, 0.03],
                                                [-(size[0]/2 - 0.07), -(size[0]/2 - 0.07), 0.03]],
                                 linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                                 linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                 linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                                 linkParentIndices=[0, 0, 0, 0, 0],
                                 linkJointTypes=[pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED],
                                 linkJointAxis=[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    )
    pb.changeDynamics(self.id, -1, lateralFriction=0.001)

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    pb.resetBasePositionAndOrientation(self.id, pos, rot)

  def getObjPose(self):
    link_state = pb.getLinkState(self.id, 2)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getPullPose(self):
    link_state = pb.getLinkState(self.id, 3)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getPressPose(self):
    link_state = pb.getLinkState(self.id, 4)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getPose(self):
    pos, rot = pb.getBasePositionAndOrientation(self.id)
    return list(pos), list(rot)

  def remove(self):
    if self.id:
      pb.removeBody(self.id)
    self.id = None
