from vrep_env import VrepEnv
from pybulet_env import PyBulletEnv

def createBlockPickingEnv(simulator_base_env, config):
  class BlockPickingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if type(parent_env) is VrepEnv:
        super(BlockPickingEnv, self).__init__()
      elif type(parent_env) is PyBulletEnv:
        super(BlockPickingEnv, self).__init__()
      else:
        raise ValueError('Bad simulator base env specified.')

    def reset(self):
      ''''''
      pass

    def _checkTermination(self):
      ''''''
      pass

  def _thunk():
    return BlockPickingEnv(config)

  return _thunk
