from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

class HouseBuilding1Env(PyBulletEnv):
  ''''''
  def __init__(self, config):
    super(HouseBuilding1Env, self).__init__(config)

  def reset(self):
    ''''''
    while True:
      super(HouseBuilding1Env, self).reset()
      try:
        self._generateShapes(constants.TRIANGLE, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.CUBE, self.num_obj-1, random_orientation=self.random_orientation)
      except Exception as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return self._checkStack(blocks+triangles) and self._checkObjUpright(triangles[0])

  def isSimValid(self):
    triangles = list(filter(lambda x: self.object_types[x] == constants.TRIANGLE, self.objects))
    return self._checkObjUpright(triangles[0]) and super(HouseBuilding1Env, self).isSimValid()

def createHouseBuilding1Env(config):
  def _thunk():
    return HouseBuilding1Env(config)
  return _thunk
