from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

def createHouseBuilding5Env(simulator_base_env, config):
  class HouseBuilding5Env(simulator_base_env):
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
      assert self.num_obj % 2 == 0
      self.prev_best = 0

    def step(self, action):
      pre_n = self.getNStackedPairs()
      self.takeAction(action)
      self.wait(100)
      obs = self._getObservation(action)
      done = self._checkTermination()
      if self.reward_type == 'dense':
        cur_n = self.getNStackedPairs()
        if cur_n > pre_n:
          reward = 1.0
        elif cur_n < pre_n:
          reward = cur_n - pre_n
        else:
          reward = 0
      else:
        reward = 1.0 if done else 0.0

      if not done:
        done = self.current_episode_steps >= self.max_steps or not self.isSimValid()
      self.current_episode_steps += 1

      return obs, reward, done

    def reset(self):
      ''''''
      while True:
        super(HouseBuilding5Env, self).reset()
        try:
          self._generateShapes(constants.CYLINDER, int(self.num_obj/2), random_orientation=self.random_orientation)
          self._generateShapes(constants.CUBE, int(self.num_obj/2), random_orientation=self.random_orientation)
        except Exception as e:
          continue
        else:
          break
      return self._getObservation()

    def getNStackedPairs(self):
      blocks = list(filter(lambda x: self.object_types[x] == constants.CUBE and self._isObjOnGround(x), self.objects))
      cylinders = list(filter(lambda x: self.object_types[x] == constants.CYLINDER, self.objects))

      n = 0

      for block in blocks:
        for cylinder in cylinders:
          if self._checkOnTop(block, cylinder):
            cylinders.remove(cylinder)
            n += 1
            break

      return n


    def _checkTermination(self):
      return self.getNStackedPairs() == int(self.num_obj/2)

    def getObjectPosition(self):
      return list(map(self._getObjectPosition, self.objects))

    def isSimValid(self):
      cylinders = list(filter(lambda x: self.object_types[x] == constants.CYLINDER, self.objects))
      for cylinder in cylinders:
        if not self._checkObjUpright(cylinder):
          return False
      return super().isSimValid()

  def _thunk():
    return HouseBuilding5Env(config)

  return _thunk