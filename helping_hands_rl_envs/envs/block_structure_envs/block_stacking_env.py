from helping_hands_rl_envs.envs.base_env import BaseEnv
from helping_hands_rl_envs.pybullet.utils import constants
from helping_hands_rl_envs.pybullet.utils.constants import NoValidPositionException

class BlockStackingEnv(BaseEnv):
  ''''''
  def __init__(self, config):
    # env specific parameters
    if 'object_scale_range' not in config:
      config['object_scale_range'] = [0.6, 0.6]
    if 'num_objects' not in config:
      config['num_objects'] = 3
    if 'max_steps' not in config:
      config['max_steps'] = 10
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
      self.resetPybulletWorkspace()
      try:
        self._generateShapes(object_type, self.num_obj, random_orientation=self.random_orientation)
      except NoValidPositionException as e:
        continue
      else:
        break
    return self._getObservation()

  def _checkTermination(self):
    ''''''
    return self._checkStack()

def createBlockStackingEnv(config):
  return BlockStackingEnv(config)
