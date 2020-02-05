import pybullet as pb

class PybulletObject(object):
  def __init__(self, object_type_id, object_id):
    self.object_type_id = object_type_id
    self.object_id = object_id

  def getXPosition(self):
    return self.getPosition()[0]

  def getYPosition(self):
    return self.getPosition()[1]

  def getXYPosition(self):
    return self.getPosition()[:2]

  def getZPosition(self):
    return self.getPosition()[2]

  def getPosition(self):
    pos, _ = pb.getBasePositionAndOrientation(self.object_id)
    return list(pos)

  def getRotation(self):
    _, rot = pb.getBasePositionAndOrientation(self.object_id)
    return list(rot)

  def getPose(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    return list(pos), list(rot)

  def getBoundingBox(self):
    return list(pb.getAABB(self.object_id))

  def getContactPoints(self):
    return pb.getContactPoints(self.object_id)

  def isTouching(self, obj):
    contact_points = self.getContactPoints()
    for p in contact_points:
      if p[2] == obj.object_id:
        return True
    return False

  def resetPose(self, pos, rot):
    pb.resetBasePositionAndOrientation(self.object_id, pos, rot)

  def __eq__(self, other):
    if not isinstance(other, PybulletObject):
      return False
    return self.object_id == other.object_id and self.object_type_id == other.object_type_id

  def __hash__(self):
    return self.object_id