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
    return pos

  def getRotation(self):
    _, rot = pb.getBasePositionAndOrientation(self.object_id)
    return rot

  def getPose(self):
    pos, rot = pb.getBasePositionAndOrientation(self.object_id)
    return pos, rot

  def resetPose(self, pos, rot):
    pb.resetBasePositionAndOrientation(self.object_id, pos, rot)
