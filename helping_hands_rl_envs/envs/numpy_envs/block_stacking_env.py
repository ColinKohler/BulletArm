from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

def createBlockStackingEnv(simulator_base_env, config):
  class BlockStackingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is NumpyEnv:
        super().__init__(config)
      elif simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')

      self.simulator_base_env = simulator_base_env
      self.object_type = config['object_type'] if 'object_type' in config else 'cube'
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'

    def step(self, action):
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination()
      reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

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

  def _thunk():
    return BlockStackingEnv(config)

  return _thunk
