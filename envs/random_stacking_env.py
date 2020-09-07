from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

def createRandomStackingEnv(simulator_base_env, config):
  class RandomStackingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      config['check_random_obj_valid'] = True
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')

      self.simulator_base_env = simulator_base_env
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
      super(RandomStackingEnv, self).reset()
      self._generateShapes(constants.RANDOM, self.num_obj, random_orientation=self.random_orientation, z_scale=1.5)
      return self._getObservation()

    def _checkTermination(self):
      ''''''
      return self._checkStack()

  def _thunk():
    return RandomStackingEnv(config)

  return _thunk
