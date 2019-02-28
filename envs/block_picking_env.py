from helping_hands_rl_envs.envs.vrep_env import VrepEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

def createBlockPickingEnv(simulator_base_env, config):
  class BlockPickingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is VrepEnv:
        super(BlockPickingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                              config['obs_size'], config['port'], config['fast_mode'])
      elif simulator_base_env is PyBulletEnv:
        super(BlockPickingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                              config['obs_size'], config['fast_mode'], config['render'])
      else:
        raise ValueError('Bad simulator base env specified.')

    def reset(self):
      ''''''
      super(BlockPickingEnv, self).reset()

      self.block = self._generateShapes(0, 1)[0]

      return self._getObservation()

    def _checkTermination(self):
      ''''''
      block_position = self._getObjectPosition(self.block)
      # print('{} > {}'.format(block_position[2], self.rest_pose[0][2] - 0.25))
      return block_position[2] > self.rest_pose[2,-1] - 0.25

  def _thunk():
    return BlockPickingEnv(config)

  return _thunk
