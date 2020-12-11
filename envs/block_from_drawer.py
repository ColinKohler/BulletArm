import numpy as np

from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from envs.pybullet_envs.two_view_envs.two_view_env import TwoViewEnv
from helping_hands_rl_envs.simulators import constants

def createBlockFromDrawerEnv(simulator_base_env, config):
  class BlockFromDrawerEnv(TwoViewEnv):
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'
      self.handle1_pos = None
      self.handle2_pos = None

    def reset(self):
      super().reset()
      if np.random.random() < 0.5:
        self._generateShapes(constants.CUBE, 1, pos=[[0.8, 0, 0]], random_orientation=self.random_orientation)
      else:
        self._generateShapes(constants.CUBE, 1, pos=[[0.8, 0, 0.2]], random_orientation=self.random_orientation)

      self.handle1_pos = self.drawer.getHandlePosition()
      self.handle2_pos = self.drawer2.getHandlePosition()
      return self._getObservation()

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

    def _checkTermination(self):
      for o in self.objects:
        if not (self._isObjOnGround(o) and self._isObjectWithinWorkspace(o)):
          return False
      return True

    def isSimValid(self):
      return True

  def _thunk():
    return BlockFromDrawerEnv(config)

  return _thunk
