import itertools

class NumpyObject(object):
  new_id = itertools.count().next

  def __init__(self, object_type_id, pos, rot, size, heightmap):
    self.id = resource_cl.new_id()
    self.object_type_id = object_type_id

    self.pos = pos
    self.rot = rot
    self.size = size
    self.height = pos[-1]

    self.heightmap = heightmap
    self.heightmap_size = heightmap.shape[0]
