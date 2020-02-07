import numpy as np

from helping_hands_rl_envs.envs.numpy_env import NumpyEnv
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

def createBlockConstructionEnv(simulator_base_env, config):
  class BlockConstructionEnv(simulator_base_env):
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
      self.min_top = self.num_obj

    def step(self, action):
      if self.reward_type == 'step_left_optimal':
        pre_step_left = self.getStepLeft()
        self.saveState()
        motion_primitive, x, y, z, rot = self._getSpecificAction(action)
        optimal_action = self.planBlockStackingWithX(motion_primitive, x, y)
        self.takeAction(optimal_action)
        self.wait(100)
        optimal_step_left = self.getStepLeft()
        self.restoreState()
      else:
        pre_step_left = 0
        optimal_step_left = 0

      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation()
      done = self._checkTermination()
      curr_num_top = self._getNumTopBlock()
      if self.reward_type == 'dense':
        if 0 < curr_num_top < self.min_top:
          reward = float(self.min_top - curr_num_top)
          self.min_top = curr_num_top
        else:
          reward = 0.0
      elif self.reward_type == 'step_left':
        reward = self.getStepLeft()
      elif self.reward_type == 'step_left_optimal':
        step_left = self.getStepLeft()
        if optimal_step_left > pre_step_left:
          optimal_step_left = pre_step_left
        reward = step_left, optimal_step_left
      else:
        reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      super(BlockConstructionEnv, self).reset()
      self.blocks = self._generateShapes(0, self.num_obj, random_orientation=self.random_orientation)
      self.min_top = self.num_obj
      return self._getObservation()

    def saveState(self):
      super(BlockConstructionEnv, self).saveState()
      self.stacking_state = {'min_top': deepcopy(self.min_top)}

    def restoreState(self):
      super(BlockConstructionEnv, self).restoreState()
      self.blocks = self.objects
      self.min_top = self.stacking_state['min_top']

    def _checkTermination(self):
      ''''''
      return self._checkStack()

    def _estimateIfXPossible(self, primitive, x, y):
      z = self._getPrimativeHeight(primitive, x, y)
      if primitive == self.PICK_PRIMATIVE:
        return self._checkPickValid(x, y, z, 0, False)
      else:
        return self._checkPlaceValid(x, y, z, 0, False)

    def getPlan(self):
      return self.planBlockStacking()

    def getStepLeft(self):
      if not self.isSimValid():
        return 100
      step_left = 2 * (self._getNumTopBlock() - 1)
      if self._isHolding():
        step_left -= 1
      return step_left

  def _thunk():
    return BlockConstructionEnv(config)

  return _thunk
