from bulletarm.envs.base_env import BaseEnv
from bulletarm.pybullet.utils import constants
from bulletarm.pybullet.utils.constants import NoValidPositionException

class BrickStackingEnv(BaseEnv):
  ''''''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
    super(BrickStackingEnv, self).__init__(config)
    self.num_cubes = self.num_obj - 1

  def reset(self):
    ''''''
    while True:
      self.resetPybulletWorkspace()
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
