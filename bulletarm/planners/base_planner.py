import numpy as np
import numpy.random as npr

from bulletarm.pybullet.utils import constants

class BasePlanner(object):
  def __init__(self, env, config):
    self.env = env
    self.rand_pick_prob = config['rand_pick_prob'] if 'rand_pick_prob' in config else 0.0
    self.rand_place_prob = config['rand_place_prob'] if 'rand_place_prob' in config else 0.0
    self.pick_noise = config['pick_noise'] if 'pick_noise' in config else None
    self.place_noise = config['place_noise'] if 'place_noise' in config else None
    self.planner_res = config['planner_res'] if 'planner_res' in config else 10
    self.rot_noise = config['rot_noise'] if 'rot_noise' in config else None
    self.pos_noise = config['pos_noise'] if 'rot_noise' in config else None
    self.gamma = config['gamma']  if 'gamma' in config else 0.9
    self.random_orientation = config['random_orientation'] if 'random_orientation' in config else True
    self.half_rotation = config['half_rotation'] if 'half_rotation' in config else False

    npr.seed(env.seed)

  def getNextAction(self):
    raise NotImplemented('Planners must implement this function')

  def getStepsLeft(self):
    raise NotImplemented('Planners must implement this function')

  def getValue(self):
    return self.gamma**self.getStepsLeft()

  def addNoiseToPos(self, x, y, primative):
    signs = [-1, 1]
    if primative == constants.PICK_PRIMATIVE and self.pick_noise:
      x_noise = np.round(npr.choice(signs) * npr.uniform(self.pick_noise[0], self.pick_noise[1]), self.planner_res)
      y_noise = np.round(npr.choice(signs) * npr.uniform(self.pick_noise[0], self.pick_noise[1]), self.planner_res)

      x = np.clip(x + x_noise, self.env.workspace[0,0], self.env.workspace[0,1])
      y = np.clip(y + y_noise, self.env.workspace[1,0], self.env.workspace[1,1])
    elif primative == constants.PLACE_PRIMATIVE and self.place_noise:
      x_noise = np.round(npr.choice(signs) * npr.uniform(self.place_noise[0], self.place_noise[1]), self.planner_res)
      y_noise = np.round(npr.choice(signs) * npr.uniform(self.place_noise[0], self.place_noise[1]), self.planner_res)

      x = np.clip(x + x_noise, self.env.workspace[0,0], self.env.workspace[0,1])
      y = np.clip(y + y_noise, self.env.workspace[1,0], self.env.workspace[1,1])

    return x, y

  def addNoiseToRot(self, rot):
    if self.rot_noise:
      rot = np.clip(rot + npr.uniform(-self.rot_noise, self.rot_noise), 0., np.pi)
    return rot

  def getRandomAction(self):
    if self.isHolding():
      return self.getRandomPlacingAction()
    else:
      return self.getRandomPickingAction()

  def getRandomPickingAction(self):
    x = npr.uniform(self.env.workspace[0, 0] + 0.025, self.env.workspace[0, 1] - 0.025)
    y = npr.uniform(self.env.workspace[1, 0] + 0.025, self.env.workspace[1, 1] - 0.025)
    z = 0.
    r = npr.uniform(0., np.pi)
    return self.encodeAction(constants.PICK_PRIMATIVE, x, y, z, r)

  def getRandomPlacingAction(self):
    x = npr.uniform(self.env.workspace[0, 0] + 0.025, self.env.workspace[0, 1] - 0.025)
    y = npr.uniform(self.env.workspace[1, 0] + 0.025, self.env.workspace[1, 1] - 0.025)
    z = 0.
    r = npr.uniform(0., np.pi)
    return self.encodeAction(constants.PLACE_PRIMATIVE, x, y, z, r)

  def encodeAction(self, primitive, x, y, z, r):
    x = np.round(x, self.planner_res)
    y = np.round(y, self.planner_res)
    x, y = self.addNoiseToPos(x, y, primitive)
    # TODO: addNoiseToRot with 3 rots
    if self.rot_noise: r = self.addNoiseToRot(r)
    if self.half_rotation:
      if not hasattr(r, '__len__'):
        while r < 0:
          r += np.pi
        while r > np.pi:
          r -= np.pi
      else:
        rz, ry, rx = r
        while rz < 0:
          rz += np.pi
          rx = -rx
          ry = -ry
        while rz > np.pi:
          rz -= np.pi
          rx = -rx
          ry = -ry
        r = rz, ry, rx
    return self.env._encodeAction(primitive, x, y, z, r)

  def getObjects(self, obj_type=None):
    if obj_type is not None:
      return list(filter(lambda x: self.env.object_types[x] == obj_type, self.env.objects))
    else:
      return list(filter(lambda x: not self.isObjectHeld(x), self.env.objects))

  def getObjectsOnTopOf(self, bottom_obj):
    return list(filter(lambda x: self.checkOnTopOf(bottom_obj, x), self.getObjects()))

  # wrapper functions for accessing env

  def areObjectsInWorkspace(self):
    return self.env.areObjectsInWorkspace()

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
