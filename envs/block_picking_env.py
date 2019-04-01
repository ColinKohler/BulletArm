from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.vrep_env import VrepEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

VALID_SIMULATORS = [NumpyEnv, VrepEnv, PyBulletEnv]

def createBlockPickingEnv(simulator_base_env, config):
  class BlockPickingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is NumpyEnv:
        super(BlockPickingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                              config['obs_size'], config['render'], config['action_sequence'])
      elif simulator_base_env is VrepEnv:
        super(BlockPickingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                              config['obs_size'], config['port'], config['fast_mode'],
                                              config['action_sequence'])
      elif simulator_base_env is PyBulletEnv:
        super(BlockPickingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                              config['obs_size'], config['fast_mode'], config['render'],
                                              config['action_sequence'])
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env

    def reset(self):
      ''''''
      super(BlockPickingEnv, self).reset()

      self.blocks = self._generateShapes(0, 1, random_orientation=False)

      return self._getObservation()

    def _checkTermination(self):
      ''''''
      for obj in self.blocks:
        if self._isObjectHeld(obj):
          return True
      return False

  def _thunk():
    return BlockPickingEnv(config)

  return _thunk
