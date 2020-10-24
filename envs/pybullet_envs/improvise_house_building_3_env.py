import time
from copy import deepcopy
import numpy.random as npr
from itertools import combinations
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException

class ImproviseHouseBuilding3Env(PyBulletEnv):
  ''''''
  def __init__(self, config):
    config['check_random_obj_valid'] = True
    super(ImproviseHouseBuilding3Env, self).__init__(config)

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
      except NoValidPositionException:
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
    return self._checkObjUpright(roofs[0]) and super(ImproviseHouseBuilding3Env, self).isSimValid()

def createImproviseHouseBuilding3Env(config):
  return ImproviseHouseBuilding3Env(config)
