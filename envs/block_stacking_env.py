from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.vrep_env import VrepEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

def createBlockStackingEnv(simulator_base_env, config):
  class BlockStackingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is NumpyEnv:
        super(BlockStackingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                               config['obs_size'], config['render'], config['action_sequence'])
      elif simulator_base_env is VrepEnv:
        super(BlockStackingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                               config['obs_size'], config['port'], config['fast_mode'],
                                               config['action_sequence'])
      elif simulator_base_env is PyBulletEnv:
        super(BlockStackingEnv, self).__init__(config['seed'], config['workspace'], config['max_steps'],
                                               config['obs_size'], config['fast_mode'], config['render'],
                                               config['action_sequence'])
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1

    def reset(self):
      ''''''
      super(BlockStackingEnv, self).reset()
      self.blocks = self._generateShapes(0, self.num_obj, random_orientation=self.random_orientation)
      return self._getObservation()

    def _checkTermination(self):
      ''''''
      top_block = self.blocks[0]
      for obj in self.blocks:
        if obj.on_top:
          top_block = obj
      count = 1
      while top_block.cube_below:
        count += 1
        top_block = top_block.cube_below
      return count == self.num_obj

  def _thunk():
    return BlockStackingEnv(config)

  return _thunk
