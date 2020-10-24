from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException

class BrickStackingEnv(PyBulletEnv):
  ''''''
  def __init__(self, config):
    super(BrickStackingEnv, self).__init__(config)
    self.num_cubes = self.num_obj - 1

  def reset(self):
    ''''''
    while True:
      super(BrickStackingEnv, self).reset()
      try:
        self._generateShapes(constants.BRICK, 1, random_orientation=self.random_orientation)
        self._generateShapes(constants.CUBE, self.num_cubes, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
    bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
    return all([self._checkOnTop(bricks[0], b) for b in blocks])

def createBrickStackingEnv(config):
  return BrickStackingEnv(config)
