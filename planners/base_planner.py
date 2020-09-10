import numpy as np
import numpy.random as npr

class BasePlanner(object):
  def __init__(self, env, config):
    self.env = env
    self.rand_pick_prob = config['rand_pick_prob'] if 'rand_pick_prob' in config else 0.0
    self.rand_place_prob = config['rand_place_prob'] if 'rand_place_prob' in config else 0.0
    self.pos_noise = config['pos_noise'] if 'pos_noise' in config else None
    self.rot_noise = config['rot_noise'] if 'rot_noise' in config else None
    self.gamma = config['gamma']  if 'gamma' in config else 0.9
    self.random_orientation = config['random_orientation'] if 'random_orientation' in config else True

  def getNextAction(self):
    raise NotImplemented('Planners must implement this function')

  def getStepLeft(self):
    raise NotImplemented('Planners must implement this function')

  def getValue(self):
    return self.gamma**self.getStepLeft()

  def addNoiseToPos(self, x, y):
    # TODO: Would we ever want to include noise on the z-axis here?
    if self.pos_noise:
      x = np.clip(x + npr.uniform(-self.pos_noise, self.pos_noise), self.env.workspace[0,0], self.env.workspace[0,1])
      y = np.clip(y + npr.uniform(-self.pos_noise, self.pos_noise), self.env.workspace[1,0], self.env.workspace[1,1])
    return x, y

  def addNoiseToRot(self, rot):
    if self.rot_noise:
      rot = np.clip(rot + npr.uniform(-self.rot_noise, self.rot_noise), 0., np.pi)
    return rot

  def encodeAction(self, primitive, x, y, z, r):
    if self.pos_noise: x, y = self.addNoiseToPos(x, y)
    if self.rot_noise: r = self.addNoiseToRot(r)
    return self.env._encodeAction(primitive, x, y, z, r)

  def getObjects(self, obj_type=None):
    if obj_type is not None:
      return list(filter(lambda x: self.env.object_types[x] == obj_type, self.env.objects))
    else:
      return self.env.objects

  def getObjectsOnTopOf(self, bottom_obj):
    return list(filter(lambda x: self.checkOnTopOf(bottom_obj, x), self.getObjects()))

  # wrapper functions for accessing env

  def getMaxBlockSize(self):
    return self.env.max_block_size

  def getDistance(self, obj1, obj2):
    position1 = obj1.getPosition()
    position2 = obj2.getPosition()
    return np.linalg.norm(np.array(position1) - np.array(position2))

  def getValidPositions(self, padding, min_distance, existing_positions, num_shapes, sample_range=None):
    return self.env._getValidPositions(padding, min_distance, existing_positions, num_shapes, sample_range)

  def getNumTopBlock(self, objects=None):
    return self.env._getNumTopBlock(objects)

  def checkOnTopOf(self, bottom_obj, top_obj):
    return self.env._checkOnTop(bottom_obj, top_obj)

  def checkInBetween(self, middle_obj, side_obj1, side_obj2):
    return self.env._checkInBetween(middle_obj, side_obj1, side_obj2)

  def checkStack(self, objects):
    return self.env._checkStack(objects)

  def checkTermination(self):
    return self.env._checkTermination()

  def isObjectHeld(self, obj):
    return self.env._isObjectHeld(obj)

  def isObjOnTop(self, obj):
    return self.env._isObjOnTop(obj)

  def isHolding(self):
    return self.env._isHolding()

  def isSimValid(self):
    return self.env.isSimValid()

  def isObjOnGround(self, obj):
    return self.env._isObjOnGround(obj)

  def getHoldingObj(self):
    return self.env._getHoldingObj()

  def getHoldingObjType(self):
    return self.env._getHoldingObjType()
