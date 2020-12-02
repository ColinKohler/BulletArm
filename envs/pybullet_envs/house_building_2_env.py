from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException

class HouseBuilding2Env(PyBulletEnv):
  ''''''
  def __init__(self, config):
    super(HouseBuilding2Env, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      super(HouseBuilding2Env, self).reset()
      try:
        self._generateShapes(constants.CUBE, 2, random_orientation=self.random_orientation)
        self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
      except NoValidPositionException:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

    top_blocks = []
    for block in blocks:
      if self._isObjOnTop(block, blocks):
        top_blocks.append(block)
    if len(top_blocks) != 2:
      return False
    if self._checkOnTop(top_blocks[0], roofs[0]) and self._checkOnTop(top_blocks[1], roofs[0]):
      return True
    return False

  def isSimValid(self):
    roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
    return self._checkObjUpright(roofs[0]) and super(HouseBuilding2Env, self).isSimValid()

def createHouseBuilding2Env(config):
  return HouseBuilding2Env(config)
