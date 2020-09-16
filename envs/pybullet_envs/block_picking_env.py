from copy import deepcopy
import numpy as np
from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

def createBlockPickingEnv(simulator_base_env, config):
  class BlockPickingEnv(simulator_base_env):
    ''''''

    def __init__(self, config):
      if simulator_base_env is NumpyEnv:
        super().__init__(config)
      elif simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'
      self.obj_grasped = 0

    def step(self, action):
      pre_obj_grasped = self.obj_grasped
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination()
      if self.reward_type == 'dense':
        reward = 1.0 if self.obj_grasped > pre_obj_grasped else 0.0
      elif self.reward_type == 'step_left':
        reward = self.getStepLeft()
      else:
        reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      super(BlockPickingEnv, self).reset()
      self._generateShapes(0, self.num_obj, random_orientation=self.random_orientation)
      self.obj_grasped = 0
      return self._getObservation()

    def saveState(self):
      super(BlockPickingEnv, self).saveState()
      self.state['obj_grasped'] = deepcopy(self.obj_grasped)

    def restoreState(self):
      super(BlockPickingEnv, self).restoreState()
      self.obj_grasped = self.state['obj_grasped']

    def getObjectPosition(self):
      return list(map(self._getObjectPosition, self.objects))

    def _checkTermination(self):
      ''''''
      for obj in self.objects:
        if self._isObjectHeld(obj):
          self.obj_grasped += 1
          self._removeObject(obj)
          if self.obj_grasped == self.num_obj:
            return True
          return False
      return False

    def _getObservation(self):
      state, in_hand, obs = super(BlockPickingEnv, self)._getObservation()
      return 0, np.zeros_like(in_hand), obs

  def _thunk():
    return BlockPickingEnv(config)

  return _thunk
