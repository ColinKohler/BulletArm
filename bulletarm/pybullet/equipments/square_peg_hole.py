import os
import pybullet as pb
import numpy as np

class SquarePegHole(object):
  '''

  '''
  def __init__(self):
    self.id = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), size=(0.2, 0.2, 0.1)):
    ''''''
    bottom_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, 0.001], rgbaColor=[0, 0, 0, 1])
    bottom_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/2, size[1]/2, 0.001])

    face_a_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/6, size[1]/2, 0.05], rgbaColor=[0, 0, 1, 1])
    face_a_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/6, size[1]/2, 0.05])

    face_b_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/6, size[1]/2, 0.05], rgbaColor=[0, 0, 1, 1])
    face_b_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/6, size[1]/2, 0.05])

    face_c_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/6, size[1]/6, 0.05], rgbaColor=[0, 0, 1, 1])
    face_c_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/6, size[1]/6, 0.05])

    face_d_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[size[0]/6, size[1]/6, 0.05], rgbaColor=[0, 0, 1, 1])
    face_d_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[size[0]/6, size[1]/6, 0.05])

    hole_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1, 0, 0, 0])
    hole_collision = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0, 0, 0], rgbaColor=[1, 0, 0, 1])

    self.id = pb.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=bottom_collision,
      baseVisualShapeIndex=bottom_visual,
      basePosition=pos,
      baseOrientation=rot,
      linkMasses=[1, 1, 1, 1, 1],
      linkCollisionShapeIndices=[face_a_collision, face_b_collision, face_c_collision, face_d_collision, hole_visual],
      linkVisualShapeIndices=[face_a_visual, face_b_visual, face_c_visual, face_d_visual, hole_collision],
      linkPositions=[[0, -size[1]/3, 0.05],
                     [0, size[1]/3, 0.05],
                     [-size[1]/3, 0,  0.05],
                     [size[1]/3, 0, 0.05],
                     [0, 0, 0.08]],
      linkOrientations=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
      linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
      linkParentIndices=[0, 0, 0, 0, 0],
      linkJointTypes=[pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED],
      linkJointAxis=[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    )

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    ''''''
    pb.resetBasePositionAndOrientation(self.id, pos, rot)

  def getPose(self):
    ''''''
    pos, rot = pb.getBasePositionAndOrientation(self.id)
    return list(pos), list(rot)

  def getHolePose(self):
    link_state = pb.getLinkState(self.id, 4)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def remove(self):
    ''''''
    if self.id:
      pb.removeBody(self.id)
    self.id = None