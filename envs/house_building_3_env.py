from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv

def createHouseBuilding3Env(simulator_base_env, config):
  class HouseBuilding3Env(simulator_base_env):
    ''''''
    def __init__(self, config):
      if simulator_base_env is PyBulletEnv:
        super().__init__(config)
        self.block_scale_range = (0.6, 0.6)
        self.min_block_size = self.block_original_size * self.block_scale_range[0]
        self.max_block_size = self.block_original_size * self.block_scale_range[1]
      else:
        raise ValueError('Bad simulator base env specified.')
      self.simulator_base_env = simulator_base_env
      self.random_orientation = config['random_orientation'] if 'random_orientation' in config else False
      self.num_obj = config['num_objects'] if 'num_objects' in config else 1
      self.reward_type = config['reward_type'] if 'reward_type' in config else 'sparse'

      self.blocks = []
      self.bricks = []
      self.roofs = []
      self.env_state = {}

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
      while True:
        super(HouseBuilding3Env, self).reset()
        try:
          self.blocks = self._generateShapes(0, 2, random_orientation=self.random_orientation)
          self.roofs = self._generateShapes(self.ROOF, 1, random_orientation=self.random_orientation)
          self.bricks = self._generateShapes(self.BRICK, 1, random_orientation=self.random_orientation)
        except:
          continue
        else:
          break
      return self._getObservation()

    def saveState(self):
      super().saveState()
      self.env_state = {'blocks': deepcopy(self.blocks),
                        'bricks': deepcopy(self.bricks),
                        'roofs': deepcopy(self.roofs)}

    def restoreState(self):
      super().restoreState()
      self.blocks = self.env_state['blocks']
      self.bricks = self.env_state['bricks']
      self.roofs = self.env_state['roofs']

    def _checkTermination(self):
      top_blocks = []
      for block in self.blocks:
        if self._isObjOnTop(block, self.blocks):
          top_blocks.append(block)
      if len(top_blocks) != 2:
        return False
      if self._checkOnTop(top_blocks[0], self.bricks[0]) and \
          self._checkOnTop(top_blocks[1], self.bricks[0]) and \
          self._checkOnTop(self.bricks[0], self.roofs[0]) and \
          self._checkOriSimilar([self.bricks[0], self.roofs[0]]):
        return True
      return False

    def getPlan(self):
      return self.planHouseBuilding3(self.blocks, self.bricks, self.roofs)

    def getObjectPosition(self):
      return list(map(self._getObjectPosition, self.objects))

    def getStepLeft(self):
      if not self.isSimValid():
        return 100
      if self._checkTermination():
        return 0
      if self.blockPosValidHouseBuilding2(self.blocks):
        if self.brickPosValidHouseBuilding3(self.blocks, self.bricks):
          step_left = 2
          if self._isObjectHeld(self.roofs[0]):
            step_left = 1
        else:
          step_left = 4
          if self._isObjectHeld(self.bricks[0]):
            step_left = 3
          elif self._isObjectHeld(self.roofs[0]):
            step_left = 5
      else:
        step_left = 6
        if self._isHolding():
          if self._isObjectHeld(self.roofs[0]) or self._isObjectHeld(self.bricks[0]):
            step_left = 7
          else:
            step_left = 5
      return step_left

    def isSimValid(self):
      return self._checkObjUpright(self.roofs[0]) and super().isSimValid()

  def _thunk():
    return HouseBuilding3Env(config)

  return _thunk