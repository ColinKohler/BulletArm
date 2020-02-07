from copy import deepcopy
from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

def createBlockCylinderStackingEnv(simulator_base_env, config):
  class BlockStackingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is NumpyEnv:
        super(BlockStackingEnv, self).__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')

      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'
      if self.num_obj % 2 != 0:
        raise ValueError('number of objects must be plural')

      self.cylinder_stacked = False
      self.block_stacked = False

    def step(self, action):
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination()
      if self.reward_type == 'dense':
        if not self.cylinder_stacked and self._getNumTopCylinder() == 1:
          self.cylinder_stacked = True
          reward = 1.0
        elif not self.block_stacked and self._getNumTopBlock() == 1:
          self.block_stacked = True
          reward = 1.0
        else:
          reward = 0.0
      else:
        reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      super(BlockStackingEnv, self).reset()
      self._generateShapes(0, int(self.num_obj/2), random_orientation=self.random_orientation)
      self._generateShapes(2, int(self.num_obj/2), random_orientation=self.random_orientation)
      self.cylinder_stacked = False
      self.block_stacked = False
      return self._getObservation()

    def saveState(self):
      super(BlockStackingEnv, self).saveState()
      self.state['cylinder_stacked'] = deepcopy(self.cylinder_stacked)
      self.state['block_stacked'] = deepcopy(self.block_stacked)

    def restoreState(self):
      super(BlockStackingEnv, self).restoreState()
      self.cylinder_stacked = self.state['cylinder_stacked']
      self.block_stacked = self.state['block_stacked']

    def _checkTermination(self):
      ''''''
      return self._getNumTopBlock() == 1 and self. _getNumTopCylinder() == 1

    def getObjectPosition(self):
      return list(map(self._getObjectPosition, self.objects))

  def _thunk():
    return BlockStackingEnv(config)

  return _thunk
