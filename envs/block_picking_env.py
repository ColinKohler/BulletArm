from envs.vrep_env import VrepEnv
from envs.pybullet_env import PyBulletEnv

def createBlockPickingEnv(simulator_base_env, config):
  class BlockPickingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is VrepEnv:
        super(BlockPickingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                              config['obs_size'], config['port'], config['fast_mode'])
      elif simulator_base_env is PyBulletEnv:
        super(BlockPickingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                              config['obs_size'], config['fast_mode'])
      else:
        raise ValueError('Bad simulator base env specified.')

    def reset(self):
      ''''''
      super(BlockPickingEnv, self).reset()

      self.blocks = self._generateShapes(0, 1)
      return self._getObservation()

    def _checkTermination(self):
      ''''''
      return False

  def _thunk():
    return BlockPickingEnv(config)

  return _thunk
