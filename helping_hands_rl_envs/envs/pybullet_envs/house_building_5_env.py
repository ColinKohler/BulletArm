from copy import deepcopy
from helping_hands_rl_envs.envs.pybullet_envs.pybullet_env import PyBulletEnv
from helping_hands_rl_envs.simulators import constants
from helping_hands_rl_envs.simulators.constants import NoValidPositionException

class HouseBuilding5Env(PyBulletEnv):
  ''''''
  def __init__(self, config):
    super(HouseBuilding5Env, self).__init__(config)
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
      self.resetPybulletEnv()
      try:
        self._generateShapes(constants.CYLINDER, int(self.num_obj/2), random_orientation=self.random_orientation)
        self._generateShapes(constants.CUBE, int(self.num_obj/2), random_orientation=self.random_orientation)
      except NoValidPositionException as e:
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
    return super(HouseBuilding5Env, self).isSimValid()

def createHouseBuilding5Env(config):
  return HouseBuilding5Env(config)
