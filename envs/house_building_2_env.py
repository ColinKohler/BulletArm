from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

def createHouseBuilding2Env(simulator_base_env, config):
  # TODO: check in between
  class HouseBuilding2Env(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'

      self.blocks = []
      self.roofs = None
      self.stacking_state = {}

    def step(self, action):
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation()
      done = self._checkTermination()
      if self.reward_type == 'step_left':
        reward = self.getStepLeft()
      else:
        reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      super(HouseBuilding2Env, self).reset()
      self.blocks = self._generateShapes(0, self.num_obj-1, random_orientation=self.random_orientation)
      self.roofs = self._generateShapes(self.ROOF, 1, random_orientation=self.random_orientation)
      return self._getObservation()

    def saveState(self):
      # TODO
      pass

    def restoreState(self):
      # TODO
      pass

    def _checkTermination(self):
      top_blocks = []
      for block in self.blocks:
        if self._isObjOnTop(block, self.blocks):
          top_blocks.append(block)
      if len(top_blocks) != 2:
        return False
      if self._checkOnTop(top_blocks[0], self.roofs[0]) and self._checkOnTop(top_blocks[1], self.roofs[0]):
        return True
      return False

    def getPlan(self):
      return self.planHouseBuilding2(self.blocks, self.roofs)

    def getStepLeft(self):
      if not self.isSimValid():
        return 100
      if self._checkTermination():
        return 0
      if self.blockPosValidHouseBuilding2(self.blocks):
        step_left = 2
        if self._isObjectHeld(self.roofs[0]):
          step_left -= 1
      else:
        step_left = 4
        if self._isHolding():
          if self._isObjectHeld(self.roofs[0]):
            step_left += 1
          else:
            step_left -= 1
      return step_left

    def isSimValid(self):
      return self._checkObjUpright(self.roofs[0]) and super().isSimValid()

  def _thunk():
    return HouseBuilding2Env(config)

  return _thunk