from copy import deepcopy
from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

class HouseBuilding4Env(BaseEnv):
  '''Open loop house building 4 task.

  The robot needs to: (1) place two cubic blocks adjacent to each other, (2) put a cuboid on
  top of the two cubic bricks, (3) put another two cubic blocks on top of the cuboid, (4) put
  a root on top of the structure.

  Args:
    config (dict): Intialization arguments for the env
  '''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 6
    if 'max_steps' not in config:
      config['max_steps'] = 20
    super(HouseBuilding4Env, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.BRICK, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.CUBE, 4, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    level1_blocks = list(filter(self._isObjOnGround, blocks))
    if len(level1_blocks) != 2:
      return False

    level2_blocks = list(set(blocks) - set(level1_blocks))
    return self._checkOnTop(level1_blocks[0], bricks[0]) and \
           self._checkOnTop(level1_blocks[1], bricks[0]) and \
           self._checkOnTop(bricks[0], level2_blocks[0]) and \
           self._checkOnTop(bricks[0], level2_blocks[1]) and \
           self._checkOnTop(level2_blocks[0], roofs[0]) and \
           self._checkOnTop(level2_blocks[1], roofs[0]) and \
           self._checkOriSimilar([bricks[0], roofs[0]]) and \
           self._checkInBetween(bricks[0], level1_blocks[0], level1_blocks[1]) and \
           self._checkInBetween(roofs[0], level2_blocks[0], level2_blocks[1]) and \
           self._checkInBetween(bricks[0], level2_blocks[0], level2_blocks[1])

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(HouseBuilding4Env, self).isSimValid()

def createHouseBuilding4Env(config):
  return HouseBuilding4Env(config)
