from copy import deepcopy
from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.vrep_env import VrepEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

def createBrickStackingEnv(simulator_base_env, config):
  class BlockStackingEnv(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super(BlockStackingEnv, self).__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'

    def step(self, action):
      self.takeAction(action)
      primitive_id = self.action_sequence.find('p')
      motion_primitive = action[primitive_id] if primitive_id != -1 else 0
      self.wait(100, motion_primitive)
      obs = self._getObservation()
      done = self._checkTermination()
      reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      super(BlockStackingEnv, self).reset()
      self.blocks = self._generateShapes(self.CUBE, 1, random_orientation=self.random_orientation)
      self.bricks = self._generateShapes(self.BRICK, 1, random_orientation=self.random_orientation)
      return self._getObservation()

    def saveState(self):
      super(BlockStackingEnv, self).saveState()
      self.stacking_state = {'blocks': deepcopy(self.blocks),
                             'bricks': deepcopy(self.bricks)}

    def restoreState(self):
      super(BlockStackingEnv, self).restoreState()
      self.blocks = self.stacking_state['blocks']
      self.bricks = self.stacking_state['bricks']

    def _checkTermination(self):
      ''''''
      return self._checkOnTop(self.blocks[0], self.bricks[0])

    def getObjectPosition(self):
      return list(map(self._getObjectPosition, self.blocks+self.bricks))

  def _thunk():
    return BlockStackingEnv(config)

  return _thunk
