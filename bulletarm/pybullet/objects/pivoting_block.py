import pybullet as pb

from bulletarm.pybullet.objects.pybullet_object import PybulletObject
from bulletarm.pybullet.utils import constants


class PivotingBlock(PybulletObject):
  def __init__(self, pos, rot, scale):
    bottom_visual = pb.createVisualShape(
      pb.GEOM_BOX,
      halfExtents=[0.05*scale, 0.075*scale, 0.015],
      rgbaColor=[1, 1, 1, 1]
    )
    bottom_collision = pb.createCollisionShape(
      pb.GEOM_BOX,
      halfExtents=[0.05*scale, 0.075*scale, 0.015]
    )
    object_id = pb.createMultiBody(baseMass=0.1,
                            baseCollisionShapeIndex=bottom_collision,
                            baseVisualShapeIndex=bottom_visual,
                            basePosition=pos,
                            baseOrientation=rot,
                            )
    pb.changeVisualShape(object_id, -1, rgbaColor=[0, 0, 1, 1])
    pb.changeDynamics(
      object_id,
      -1,
      lateralFriction=5.0,
    )
    super().__init__(constants.PIVOTING_BLOCK, object_id)
