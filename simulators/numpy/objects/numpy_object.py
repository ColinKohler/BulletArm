import itertools

class NumpyObject(object):
  id_iter = itertools.count()

  def __init__(self, object_type_id, pos, rot, size):
    self.id = next(NumpyObject.id_iter)
    self.object_type_id = object_type_id

    self.pos = pos
    self.rot = rot
    self.size = size
    self.height = pos[-1]

  def getPosition(self):
    return self.pos

  def getRotation(self):
    return [0., 0., self.rot]

  def getPose(self):
    return self.getPosition(), self.getRotation()
