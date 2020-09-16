from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

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
        super(HouseBuilding2Env, self).reset()
        try:
          self._generateShapes(constants.CUBE, 2, random_orientation=self.random_orientation)
          self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
        except:
          continue
        else:
          break
      return self._getObservation()

    def _checkTermination(self):
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))

      top_blocks = []
      for block in blocks:
        if self._isObjOnTop(block, blocks):
          top_blocks.append(block)
      if len(top_blocks) != 2:
        return False
      if self._checkOnTop(top_blocks[0], roofs[0]) and self._checkOnTop(top_blocks[1], roofs[0]):
        return True
      return False

    def isSimValid(self):
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkObjUpright(roofs[0]) and super().isSimValid()

  def _thunk():
    return HouseBuilding2Env(config)

  return _thunk