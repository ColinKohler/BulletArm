import time
from copy import deepcopy
import numpy.random as npr
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants

def createImproviseHouseBuilding3Env(simulator_base_env, config):
  class ImproviseHouseBuilding3Env(simulator_base_env):
    ''''''
    def __init__(self, config):
      config['check_random_obj_valid'] = True
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
        super(ImproviseHouseBuilding3Env, self).reset()
        try:
          self._generateShapes(constants.ROOF, 1, random_orientation=self.random_orientation)
          for i in range(self.num_obj-1):
            self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation, z_scale=npr.choice([1, 2], p=[0.75, 0.25]))
          # self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation,
          #                      z_scale=1)
          # self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation,
          #                      z_scale=1)
          # self._generateShapes(constants.RANDOM, 1, random_orientation=self.random_orientation,
          #                      z_scale=2)
        except:
          continue
        else:
          break
      return self._getObservation()

    def _checkTermination(self):
      rand_objs = list(filter(lambda x: self.object_types[x] == constants.RANDOM, self.objects))
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      if roofs[0].getZPosition() < 1.4*self.min_block_size:
        return False

      rand_obj_combs = combinations(rand_objs, 2)
      for (obj1, obj2) in rand_obj_combs:
        if self._checkOnTop(obj1, roofs[0]) and self._checkOnTop(obj2, roofs[0]):
          return True
      return False

    def isSimValid(self):
      roofs = list(filter(lambda x: self.object_types[x] == constants.ROOF, self.objects))
      return self._checkObjUpright(roofs[0]) and super().isSimValid()

  def _thunk():
    return ImproviseHouseBuilding3Env(config)

  return _thunk