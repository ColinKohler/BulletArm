import pybullet as pb
import numpy as np

class Pivot(object):
  '''

  '''
  def __init__(self):
    self.id = None

  def initialize(self, pos=(0,0,0), rot=(0,0,0,1), size=(0.15, 0.02, 0.04)):
    ''''''
    self.size = size

    visual = pb.createVisualShape(
      pb.GEOM_BOX,
      halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2],
      rgbaColor=[0,0,0,1]
    )
    collision = pb.createCollisionShape(
      pb.GEOM_BOX,
      halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2],
    )

    pivoting_block_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.0, 0.0, 0.0], rgbaColor=[1,0,0,1])
    pivoting_block_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0, 0, 0])

    pivoting_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1,0,0,1])
    pivoting_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0, 0, 0])

    lift_1_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1,0,0,1])
    lift_1_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0, 0, 0])

    lift_2_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1,0,0,1])
    lift_2_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0, 0, 0])

    lift_3_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1,0,0,1])
    lift_3_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0, 0, 0])

    lift_4_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1,0,0,1])
    lift_4_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0, 0, 0])

    lift_5_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1,0,0,1])
    lift_5_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0, 0, 0])

    lift_6_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1,0,0,1])
    lift_6_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0, 0, 0])

    lift_7_visual = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[0.005, 0.005, 0.005], rgbaColor=[1,0,0,1])
    lift_7_collision = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=[0, 0, 0])

    self.id = pb.createMultiBody(
      baseMass=0,
      baseCollisionShapeIndex=collision,
      baseVisualShapeIndex=visual,
      basePosition=pos,
      baseOrientation=rot,
      linkMasses=[0, 0, 0, 0, 0, 0, 0, 0, 0],
      linkCollisionShapeIndices=[pivoting_block_collision, pivoting_collision, lift_1_collision, lift_2_collision, lift_3_collision, lift_4_collision, lift_5_collision, lift_6_collision, lift_7_collision],
      linkVisualShapeIndices=[pivoting_block_visual, pivoting_visual, lift_1_visual, lift_2_visual, lift_3_visual, lift_4_visual, lift_5_visual, lift_6_visual, lift_7_visual],
      linkPositions=[[0, 0.0875, 0.02], [0, 0.175, -0.01], [0, 0.170, -0.01], [0, 0.1675, 0.0], [0, 0.165, 0.02], [0, 0.1625, 0.03], [0, 0.150, 0.04], [0, 0.145, 0.05], [0, 0.140, 0.06]],
      linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
      linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
      linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],  [0, 0, 0, 1], [0, 0, 0, 1],  [0, 0, 0, 1], [0, 0, 0, 1],  [0, 0, 0, 1]],
      linkParentIndices=[0, 0, 0, 0, 0, 0, 0, 0, 0],
      linkJointTypes=[pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED, pb.JOINT_FIXED],
      linkJointAxis=[[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]
    )
    pb.changeDynamics(self.id, -1, lateralFriction=5.0)

  def reset(self, pos=(0,0,0), rot=(0,0,0,1)):
    ''''''
    pb.resetBasePositionAndOrientation(self.id, pos, rot)

  def getPose(self):
    ''''''
    return list(self.pos), list(self.rot)

  def getPivotingBlockPose(self):
    link_state = pb.getLinkState(self.id, 0)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getPivotingPose(self):
    link_state = pb.getLinkState(self.id, 1)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getLift1Pose(self):
    link_state = pb.getLinkState(self.id, 2)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getLift2Pose(self):
    link_state = pb.getLinkState(self.id, 3)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getLift3Pose(self):
    link_state = pb.getLinkState(self.id, 4)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getLift4Pose(self):
    link_state = pb.getLinkState(self.id, 5)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getLift5Pose(self):
    link_state = pb.getLinkState(self.id, 6)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getLift6Pose(self):
    link_state = pb.getLinkState(self.id, 7)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)

  def getLift7Pose(self):
    link_state = pb.getLinkState(self.id, 8)
    pos, rot = link_state[0], link_state[1]
    return list(pos), list(rot)


  def remove(self):
    ''''''
    if self.id:
      pb.removeBody(self.id)
    self.id = None
