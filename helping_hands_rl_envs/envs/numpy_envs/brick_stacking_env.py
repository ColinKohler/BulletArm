from copy import deepcopy

from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

from helping_hands_rl_envs.simulators import constants

def createBrickStackingEnv(simulator_base_env, config):
  class BrickStackingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super(BrickStackingEnv, self).__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'
      self.num_cubes = config['num_cubes'] if 'num_cubes' in config else 2

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
      while True:
        super(BrickStackingEnv, self).reset()
        try:
          self._generateShapes(constants.BRICK, 1, random_orientation=self.random_orientation)
          self._generateShapes(constants.CUBE, self.num_cubes, random_orientation=self.random_orientation)
        except Exception as e:
          continue
        else:
          break
      return self._getObservation()

    def _checkTermination(self):
      ''''''
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      bricks = list(filter(lambda x: self.object_types[x] == constants.BRICK, self.objects))
      return all([self._checkOnTop(bricks[0], b) for b in blocks])

    def getObjectPosition(self):
      return list(map(self._getObjectPosition, self.objects))

  def _thunk():
    return BrickStackingEnv(config)

  return _thunk
