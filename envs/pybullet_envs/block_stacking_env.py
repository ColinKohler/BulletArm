from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

class BlockStackingEnv(PyBulletEnv):
  ''''''
  def __init__(self, config):
    super(BlockStackingEnv, self).__init__(config)

  def reset(self):
    ''''''
    # TODO: Move this to a utils file somewhere and set this in the init fn
    if self.object_type == 'cube':
      object_type = constants.CUBE
    elif self.object_type == 'cylinder':
      object_type = constants.CYLINDER
    else:
      raise ValueError('Invalid object type specified. Must be \'cube\' or \'cylinder\'')

    while True:
      super(BlockStackingEnv, self).reset()
      try:
        self._generateShapes(object_type, self.num_obj, random_orientation=self.random_orientation)
      except Exception as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    return self._checkStack()

def createBlockStackingEnv(config):
  def _thunk():
    return BlockStackingEnv(config)
  return _thunk
